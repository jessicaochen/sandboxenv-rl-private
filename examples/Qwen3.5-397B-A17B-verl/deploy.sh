#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Deploy / follow / tear down the Qwen3.5-397B-A17B SWE recipe.
#
#   ./deploy.sh up                      # 397b-v1: 8 trainer + 8 rollout nodes, Qwen3.5-397B-A17B, 1,024 x 16 per step
#   PROFILE=35b-easy ./deploy.sh up     # 1 + 1 nodes, Qwen3.5-35B-A3B, same code paths (profiles/*.env)
#   ./deploy.sh render             # print the rendered manifests (no cluster access)
#   ./deploy.sh logs
#   ./deploy.sh down               # Jobs, Service, ConfigMaps and leftover environments
#
# Environment (nothing cluster-specific is baked in):
#   PROFILE           profiles/<PROFILE>.env shipped to every pod as a ConfigMap (default 397b-v1)
#   PROFILE_FILE      path to a profile outside profiles/ (a copy of one with your own node counts or knobs)
#   JOB_SUFFIX        optional; suffixes the Job, Service, ConfigMap names, labels and environment ids so several
#                     instances can share a namespace (JOB_SUFFIX=a ./deploy.sh up; same value for logs/down)
#   KUBE_CONTEXT      kube context, or PROJECT_ID + ZONE + CLUSTER
#   GPU_POOL          optional node-pool label value for the GPU pods (default: any GPU node)
#   GPU_TOLERATIONS   optional extra NoSchedule taints on GPU nodes, "key=value,key2=value2"
#   GPU_POD_ENV       optional extra env for the GPU pods, "NCCL_SOCKET_IFNAME=eth0,GLOO_SOCKET_IFNAME=eth0"
#   IMAGES_JSON       tag -> digest map from integrations/uni-agent/scripts/resolve_images.sh (default ./images.json)
#   KUBECTL_ATE       kubectl-ate binary; `up`/`down` delete leftover environments (actors ua-*)
#   HF_TOKEN          optional, forwarded to the pods for model/dataset downloads
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PROFILE="${PROFILE:-397b-v1}"
PROFILE_FILE="${PROFILE_FILE:-$HERE/profiles/$PROFILE.env}"
[ -f "$PROFILE_FILE" ] || { echo "no profile at $PROFILE_FILE (PROFILE=$PROFILE)"; exit 2; }
JOB="verl-qwen3p5-397b-swe${JOB_SUFFIX:+-$JOB_SUFFIX}"
HEAD_SVC="verl-ray-head${JOB_SUFFIX:+-$JOB_SUFFIX}"
ACTOR_PREFIX="ua${JOB_SUFFIX:+-$JOB_SUFFIX}"
IMAGES_JSON="${IMAGES_JSON:-$HERE/images.json}"

render_job() {  # profile-derived sizes and GPU placement knobs into the Job manifest
  local sel="" tol="" extra="" kv nnodes_train nnodes_rollout mem eph
  nnodes_train=$(profile_get NNODES_TRAIN); nnodes_rollout=$(profile_get NNODES_ROLLOUT)
  mem=$(profile_get MEM_REQUEST); eph=$(profile_get EPH_REQUEST)
  [ -n "$nnodes_train" ] && [ -n "$nnodes_rollout" ] && [ -n "$mem" ] && [ -n "$eph" ] || { echo "$PROFILE_FILE must set NNODES_TRAIN, NNODES_ROLLOUT, MEM_REQUEST, EPH_REQUEST"; exit 2; }
  [ -n "${GPU_POOL:-}" ] && sel="cloud.google.com/gke-nodepool: ${GPU_POOL}"
  if [ -n "${GPU_TOLERATIONS:-}" ]; then
    for kv in ${GPU_TOLERATIONS//,/ }; do
      tol="${tol}${tol:+\n      }- {key: ${kv%%=*}, operator: Equal, value: \"${kv#*=}\", effect: NoSchedule}"
    done
  fi
  if [ -n "${GPU_POD_ENV:-}" ]; then
    for kv in ${GPU_POD_ENV//,/ }; do
      extra="${extra}${extra:+\n        }- {name: ${kv%%=*}, value: \"${kv#*=}\"}"
    done
  fi
  sed -e "s|__GPU_NODE_SELECTOR__|${sel}|" -e "s|__GPU_EXTRA_TOLERATIONS__|${tol}|" -e "s|__GPU_EXTRA_ENV__|${extra}|" \
      -e "s|__JOB__|${JOB}|g" -e "s|__HEAD_SVC__|${HEAD_SVC}|g" -e "s|__ACTOR_PREFIX__|${ACTOR_PREFIX}|g" \
      -e "s|__WORKER_REPLICAS__|$((nnodes_train + nnodes_rollout - 1))|" \
      -e "s|__MEM_REQUEST__|${mem}|" -e "s|__EPH_REQUEST__|${eph}|" "$HERE/job-fully-async.yaml"
}
profile_get() { sed -n "s/^$1=//p" "$PROFILE_FILE" | tail -1; }
kube_context() {
  if [ -z "${KUBE_CONTEXT:-}" ]; then
    : "${PROJECT_ID:?set KUBE_CONTEXT, or PROJECT_ID+ZONE+CLUSTER}"; : "${ZONE:?set ZONE}"; : "${CLUSTER:?set CLUSTER}"
    KUBE_CONTEXT="gke_${PROJECT_ID}_${ZONE}_${CLUSTER}"
  fi
  export KUBE_CONTEXT
  K="kubectl --context=$KUBE_CONTEXT"
}
purge_envs() {  # environments the provider left behind (sessions cut off mid-episode)
  if [ -n "${KUBECTL_ATE:-}" ]; then
    # only this instance's ids (<prefix>-<12 hex>, so `ua-` does not match `ua-a-...`) plus anything CRASHED; skip actors already
    # DELETING: their delete is in progress (or stuck on a node that no longer exists) and re-issuing it only waits
    for a in $("$KUBECTL_ATE" --context "$KUBE_CONTEXT" get actors -a ate-env 2>/dev/null | awk -v p="^${ACTOR_PREFIX}-[0-9a-f]{12}$" 'NR>1 && $4 != "ACTOR_STATE_DELETING" && ($2 ~ p || $4 == "ACTOR_STATE_CRASHED") {print $2}'); do
      "$KUBECTL_ATE" --context "$KUBE_CONTEXT" delete actor "$a" -a ate-env --any-state >/dev/null 2>&1 || true
    done
  else
    echo "note: set KUBECTL_ATE=<path to kubectl-ate> to delete leftover ${ACTOR_PREFIX}-* environments in atespace ate-env"
  fi
}
check_substrate() {
  $K -n ate-env get deploy ate-env-api >/dev/null 2>&1 || { echo "ate-env-api not found in namespace ate-env: install Substrate + ate-env first (top-level README, Quickstart)"; exit 1; }
  $K -n ate-env rollout status deploy/ate-env-api --timeout=120s
  [ -s "$IMAGES_JSON" ] || { echo "missing $IMAGES_JSON: run uni_agent_substrate.datasets --images-out and integrations/uni-agent/scripts/resolve_images.sh first"; exit 1; }
}
up_job() {  # ConfigMaps: run script, task config, image map, the code the pods install, and the profile
  local tmp; tmp=$(mktemp -d); trap 'rm -rf "$tmp"' RETURN
  tar -C "$REPO/integrations" -czf "$tmp/uni-agent-substrate.tgz" uni-agent
  tar -C "$REPO/clients" --exclude=build --exclude='*.egg-info' --exclude=__pycache__ --exclude=.venv -czf "$tmp/ate-client.tgz" python   # the ate_env client: no run-time GitHub access in the pods
  gzip -9 -c "$IMAGES_JSON" > "$tmp/images.json.gz"     # thousands of images exceed the 1 MiB ConfigMap limit uncompressed
  $K -n default create configmap "$JOB" \
      --from-file=run_fully_async.sh="$HERE/run_fully_async.sh" --from-file=task_config.yaml="$HERE/task_config.yaml" \
      --from-file=images.json.gz="$tmp/images.json.gz" \
      --from-file=uni-agent-substrate.tgz="$tmp/uni-agent-substrate.tgz" \
      --from-file=ate-client.tgz="$tmp/ate-client.tgz" \
      --from-file=offline-exclusions.txt="$REPO/benchmarking/swebench/offline-exclusions.txt" \
      --dry-run=client -o yaml | $K -n default apply --server-side --force-conflicts -f -   # server-side: no last-applied annotation (256 KiB cap) for the big payload
  $K -n default create configmap "$JOB-profile" --from-env-file="$PROFILE_FILE" --dry-run=client -o yaml | $K -n default apply -f -
  if [ -n "${HF_TOKEN:-}" ]; then
    $K -n default create secret generic hf-token --from-literal=token="$HF_TOKEN" --dry-run=client -o yaml | $K -n default apply -f -
  fi
  render_job | $K -n default apply -f -
}
down() {
  $K -n default delete job "$JOB" "$JOB-worker" --ignore-not-found --wait=true
  $K -n default delete service "$HEAD_SVC" --ignore-not-found
  $K -n default delete configmap "$JOB" "$JOB-profile" --ignore-not-found
  purge_envs
}
case "${1:-up}" in
  render) render_job ;;
  up)   kube_context; check_substrate; purge_envs; up_job
        echo "started $JOB (profile $PROFILE: $(profile_get MODEL_ID), $(profile_get NNODES_TRAIN) trainer + $(profile_get NNODES_ROLLOUT) rollout nodes); follow with: $0 logs" ;;
  logs) kube_context; $K -n default logs -f "job/$JOB" ;;
  down) kube_context; down ;;
  *) echo "usage: [PROFILE=35b-easy|35b-v1|397b-smoke|397b-easy|397b-v1] [JOB_SUFFIX=x] $0 [up|render|logs|down]"; exit 2 ;;
esac
