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
# Runs in every pod of job-fully-async.yaml (one head + NNODES_TRAIN + NNODES_ROLLOUT - 1 workers).
#   ROLE=worker : install, download the model, join the Ray head, block until the head stops Ray.
#   ROLE=head   : install, download the model, build the datasets, probe ate-env-api, start Ray, wait for
#                 all GPU nodes, then train with verl v1 separate_async:
#                 Megatron trainer across NNODES_TRAIN whole nodes (TP/CP/EP inside a node, PP across nodes,
#                 params + optimizer offloaded to CPU) and FP8 vLLM replicas on NNODES_ROLLOUT other nodes,
#                 uni-agent ReAct agent whose tools and reward executor run in Substrate environments.
# Every knob below is overridable from the pod env (deploy.sh ships profiles/<PROFILE>.env as a ConfigMap).
set -euo pipefail
export PYTHONUNBUFFERED=1 HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1 CUDA_DEVICE_MAX_CONNECTIONS=1 VLLM_USE_V1=1 \
       VLLM_ALLREDUCE_USE_SYMM_MEM=0 VLLM_DISABLE_COMPILE_CACHE=1 TORCH_NCCL_AVOID_RECORD_STREAMS=1 HYDRA_FULL_ERROR=1
: "${ROLE:=head}"; : "${NODE_WAIT_S:=3600}"; : "${RAY_HEAD_HOST:=verl-ray-head.default.svc}"; : "${RAY_PORT:=6379}"
: "${UNI_AGENT_REF:=174b040}"                                     # uni-agent commit; its verl submodule is the verl pin (3efe38c7)
: "${MODEL_ID:=Qwen/Qwen3.5-397B-A17B}"
: "${NNODES_TRAIN:=8}"; : "${NNODES_ROLLOUT:=8}"; NNODES=$((NNODES_TRAIN + NNODES_ROLLOUT))
: "${TRAIN_SOURCE:=r2e_gym}"; : "${TRAIN_DATASET:=R2E-Gym/R2E-Gym-V1}"; : "${TRAIN_SPLIT:=train}"; : "${TRAIN_SEED:=0}"; : "${TRAIN_INSTANCES:=256}"
: "${VAL_TASKS:=20}"
: "${TRAIN_BATCH:=32}"; : "${ROLLOUT_N:=8}"; : "${MINI_BATCH:=16}"; : "${SYNC_STEP:=2}"; : "${TOTAL_STEPS:=2}"
: "${SESSIONS:=96}"; : "${GATEWAYS:=4}"; : "${AGENT_WORKERS:=4}"; : "${AGENT_MAX_STEPS:=150}"
: "${MASK_UNFINISHED_EPISODE:=True}"  # True: episodes that hit AGENT_MAX_STEPS/RESPONSE_LENGTH keep their reward in the GRPO
                                      # group but train on no tokens; needs a budget most episodes finish within
: "${PROMPT_LENGTH:=8192}"; : "${RESPONSE_LENGTH:=65536}"
: "${TRAIN_TP:=2}"; : "${TRAIN_CP:=1}"; : "${TRAIN_PP:=8}"; : "${TRAIN_EP:=8}"; : "${PP_EDGE_LAYERS:=}"; : "${PARAM_OFFLOAD:=True}"
: "${ROLLOUT_TP:=8}"; : "${ROLLOUT_DP:=1}"; : "${ROLLOUT_EP:=1}"; : "${ROLLOUT_QUANT:=fp8}"
: "${ROLLOUT_GPU_MEM_UTIL:=0.85}"; : "${ROLLOUT_MAX_BATCHED_TOKENS:=16384}"; : "${ROLLOUT_MAX_NUM_SEQS:=256}"
: "${SANDBOX_STARTUP_CONCURRENCY:=32}"  # job-wide gate on environment start-ups (provider coordinator)
: "${SANDBOX_START_WAIT_S:=1800}"       # how long a session waits for sandbox capacity before failing (provider queueing)
: "${SESSION_TIMEOUT_S:=3600}"          # uni-agent session budget: capacity wait + episode + grading
: "${ROLLOUT_HYBRID_FLAGS:=1}"           # vLLM flags for the hybrid (GDN) model; set 0 if your vLLM rejects them
: "${ROLLOUT_DISABLE_CUSTOM_ALLREDUCE:=True}"  # NCCL all-reduce inside a replica; vLLM's IPC kernel can fail at start ("custom_all_reduce.cuh ... invalid argument")
: "${RAY_CPUS:=64}"                      # Ray CPU slots per node: uni-agent task runners are Ray tasks (1 CPU each)
: "${ATE_ENV_ENDPOINT:=ate-env-api.ate-env.svc:7777}"; : "${ATE_TEMPLATE:=swe-2c6g}"; : "${ACTOR_PREFIX:=ua}"
MAX_MODEL_LEN=$((PROMPT_LENGTH + RESPONSE_LENGTH))
mkdir -p /workspace/results /workspace/agent /workspace/data
cd /workspace/results   # neutral cwd: the image's working directory holds a docker/ folder that shadows the docker package

echo "=== Phase 0: topology checks (train ${NNODES_TRAIN}x8 TP$TRAIN_TP CP$TRAIN_CP PP$TRAIN_PP EP$TRAIN_EP | rollout ${NNODES_ROLLOUT}x8 TP$ROLLOUT_TP DP$ROLLOUT_DP EP$ROLLOUT_EP ${ROLLOUT_QUANT:-bf16}) ==="
[ $((SYNC_STEP * MINI_BATCH)) -eq "$TRAIN_BATCH" ] || { echo "TRAIN_BATCH must equal SYNC_STEP * MINI_BATCH"; exit 2; }
[ "$MASK_UNFINISHED_EPISODE" = True ] || [ "$MASK_UNFINISHED_EPISODE" = False ] || { echo "MASK_UNFINISHED_EPISODE must be True or False"; exit 2; }
[ $((TRAIN_TP * TRAIN_CP)) -le 8 ] && [ $((8 % (TRAIN_TP * TRAIN_CP))) -eq 0 ] || { echo "TRAIN_TP*TRAIN_CP must divide the 8 GPUs of one node (no RDMA: TP/CP stay inside a node)"; exit 2; }
[ $(((NNODES_TRAIN * 8) % TRAIN_PP)) -eq 0 ] || { echo "NNODES_TRAIN*8 must be divisible by TRAIN_PP"; exit 2; }
GPUS_PER_STAGE=$((NNODES_TRAIN * 8 / TRAIN_PP))
[ $((GPUS_PER_STAGE % (TRAIN_TP * TRAIN_CP))) -eq 0 ] && [ $((GPUS_PER_STAGE % TRAIN_EP)) -eq 0 ] && [ "$TRAIN_EP" -le 8 ] || \
  { echo "each pipeline stage has $GPUS_PER_STAGE GPUs: it must be a multiple of TRAIN_TP*TRAIN_CP and of TRAIN_EP (EP <= 8 keeps expert all-to-all inside a node)"; exit 2; }
ROLLOUT_WORLD=$((ROLLOUT_TP * ROLLOUT_DP))
{ [ "$ROLLOUT_WORLD" -le 8 ] && [ $((8 % ROLLOUT_WORLD)) -eq 0 ]; } || [ $((ROLLOUT_WORLD % 8)) -eq 0 ] || { echo "ROLLOUT_TP*ROLLOUT_DP must divide 8 or be a multiple of 8"; exit 2; }
[ $(((NNODES_ROLLOUT * 8) % ROLLOUT_WORLD)) -eq 0 ] || { echo "NNODES_ROLLOUT*8 must be divisible by ROLLOUT_TP*ROLLOUT_DP"; exit 2; }
[ "$ROLLOUT_EP" -eq 1 ] || [ "$ROLLOUT_EP" -eq "$ROLLOUT_WORLD" ] || { echo "ROLLOUT_EP must be 1 or ROLLOUT_TP*ROLLOUT_DP (verl constraint)"; exit 2; }
REPLICAS=$((NNODES_ROLLOUT * 8 / ROLLOUT_WORLD))
echo "pipeline stages: $TRAIN_PP x $GPUS_PER_STAGE GPUs; vLLM replicas: $REPLICAS x $ROLLOUT_WORLD GPUs; sessions: $SESSIONS ($((SESSIONS / REPLICAS)) per replica; beyond sandbox-pool capacity they queue up to ${SANDBOX_START_WAIT_S}s)"

pipq() {  # quiet pip install; the whole log only when it fails (the last line alone once hid git's "could not read Username")
  local log; log=$(mktemp)
  if pip install -q "$@" >"$log" 2>&1; then tail -1 "$log"; rm -f "$log"; else cat "$log"; rm -f "$log"; echo "pip install failed: $*"; exit 3; fi
}
echo "=== Phase 1: install (uni-agent@$UNI_AGENT_REF + its verl submodule, Substrate client + provider) ==="
pip freeze 2>/dev/null | grep -iE '^(torch|torchvision|torchaudio|numpy|vllm|transformers|ray|megatron|triton|flash[-_]attn|transformer[-_]engine|nvidia-|cuda-)' > /workspace/pins.txt || true
git clone -q --filter=blob:none https://github.com/verl-project/uni-agent /workspace/uni-agent
( cd /workspace/uni-agent && git checkout -q "$UNI_AGENT_REF" && git submodule update --init --depth 1 verl 2>&1 | tail -1 && echo "verl submodule: $(git -C verl rev-parse --short HEAD)" )
pipq --no-deps -e /workspace/uni-agent/verl
pipq -c /workspace/pins.txt --ignore-requires-python -e /workspace/uni-agent pydantic openai pyyaml httpx jsonschema fastapi
pipq -c /workspace/pins.txt flash-linear-attention cupy-cuda12x datasets pyarrow
# nvidia-modelopt (imported by the image's Megatron-Bridge) and swebench pull dependency trees that a
# full resolve would break the image over (it uninstalls pinned wheels): install them without deps,
# then their leaf dependencies under the image's pins.
pipq --no-deps nvidia-modelopt "swebench==4.1.0"
pipq -c /workspace/pins.txt pulp scipy docker beautifulsoup4 ghapi unidiff tenacity gitpython chardet python-dotenv rich modal   # docker: the image ships a partial module without docker.errors
python3 - <<'PY'
import importlib, subprocess, sys, traceback
PIP = {"modelopt": "nvidia-modelopt", "fla": "flash-linear-attention", "cupy": "cupy-cuda12x", "yaml": "pyyaml", "PIL": "pillow",
       "sklearn": "scikit-learn", "bs4": "beautifulsoup4", "git": "gitpython", "dotenv": "python-dotenv", "dateutil": "python-dateutil"}
for mod in ("megatron.bridge.models.conversion.auto_bridge", "swebench.harness.grading", "swebench.harness.test_spec.python",
            "fla", "cupy", "uni_agent.agents.react.agent", "verl.trainer.main_ppo"):
    tried = set()
    while True:
        try:
            importlib.import_module(mod); print("import ok:", mod, flush=True); break
        except ModuleNotFoundError as e:
            missing = (e.name or "").split(".")[0]
            pkg = PIP.get(missing, missing)
            if not missing or missing == mod.split(".")[0] or pkg in tried:
                traceback.print_exc(); raise SystemExit(f"cannot satisfy imports for {mod} (missing {missing!r})")
            tried.add(pkg); print(f"{mod}: installing missing leaf {pkg}", flush=True)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-c", "/workspace/pins.txt", pkg])
PY
mkdir -p /workspace/ate-client && tar -C /workspace/ate-client -xzf /workspace/cfg/ate-client.tgz && pipq /workspace/ate-client/python   # ate_env client from this checkout (deploy.sh payload)
tar -C /workspace/agent -xzf /workspace/cfg/uni-agent-substrate.tgz && pipq --no-deps -e /workspace/agent/uni-agent
echo "import uni_agent_substrate._register" > "$(python3 -c 'import site; print(site.getsitepackages()[0])')/uni_agent_substrate.pth"
if [ -n "$ROLLOUT_QUANT" ]; then
  # verl's online FP8 rollout setup reads hf_config.num_hidden_layers, which Qwen3.5's nested config only
  # exposes under text_config. One-line fix against the pinned verl commit; fails loudly if the line moved.
  SERVER=/workspace/uni-agent/verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py
  OLD='range(self.model_config.hf_config.num_hidden_layers)'
  NEW='range(getattr(self.model_config.hf_config, "text_config", self.model_config.hf_config).num_hidden_layers)'
  grep -qF "$OLD" "$SERVER" || grep -qF "$NEW" "$SERVER" || { echo "verl FP8 setup changed; refusing to patch $SERVER"; exit 3; }
  sed -i "s|$OLD|$NEW|" "$SERVER" && echo "patched FP8 layer-count lookup for nested (text_config) model configs"
  # Same gap one layer down: the FP8 refit finds a parameter's module by walking its HF name, but VL-style
  # models (Qwen3.5: model.language_model.* vs vLLM's language_model.model.*) need vLLM's own name mapper
  # first; otherwise nothing is quantized and the synced weights are garbage.
  python3 - <<'PY'
import pathlib
f = pathlib.Path("/workspace/uni-agent/verl/verl/utils/vllm/vllm_quant_utils.py"); s = f.read_text()
anchor = "def get_module_from_param_name(model, name: str):\n"
fix = anchor + ('    mapper = getattr(model, "hf_to_vllm_mapper", None)\n'
                '    if mapper is not None:\n'
                '        name = mapper._map_name(name) or name\n')
if fix in s:
    print("FP8 module lookup already mapper-aware")
elif anchor in s:
    s = s.replace(anchor, fix, 1); print("patched FP8 module lookup to use vLLM's hf_to_vllm_mapper")
else:
    raise SystemExit("verl FP8 module lookup changed; refusing to patch " + str(f))
# Fused (HF v5) expert tensors [E, out, in] reach the quantizer as 3-D; emit per-expert 2-D weights and
# scales named like the official FP8 checkpoints (experts.{e}.{gate,up,down}_proj.weight[_scale_inv]).
call = ("            param_lp, param_scale = scaled_fp8_blockwise(\n"
        "                v.to(dtype),\n"
        "                weight_block_size=quant_config.weight_block_size,\n"
        "            )\n")
fused = ('            if v.dim() == 3 and k.rsplit(".", 1)[-1] in ("gate_up_proj", "down_proj"):\n'
         '                base, proj = k.rsplit(".", 1)\n'
         '                for e in range(v.shape[0]):\n'
         '                    slices = {"down_proj": v[e]} if proj == "down_proj" else dict(zip(("gate_proj", "up_proj"), v[e].chunk(2, dim=0)))\n'
         '                    for sub, w in slices.items():\n'
         '                        lp, sc = scaled_fp8_blockwise(w.to(dtype).contiguous(), weight_block_size=quant_config.weight_block_size)\n'
         '                        yield (f"{base}.{e}.{sub}.weight", lp)\n'
         '                        yield (f"{base}.{e}.{sub}.weight_scale_inv", sc.squeeze(-1))\n'
         '                del v\n'
         '                continue\n') + call
if fused in s:
    print("FP8 quantizer already splits fused experts")
elif s.count(call) == 1:
    s = s.replace(call, fused, 1); print("patched FP8 quantizer to split fused expert tensors per expert")
else:
    raise SystemExit("verl FP8 quantizer changed; refusing to patch " + str(f))
f.write_text(s)
PY
fi
# A sub-batch whose samples are all loss-masked (every episode unfinished under MASK_UNFINISHED_EPISODE=True,
# typical for small smoke batches with a tight budget) must not crash the trainer: verl's rollout-correction step
# raises "response_mask must contain at least one valid token" and the actor would divide 0 by 0 tokens (NaN
# weights). Two one-line guards against the pinned verl: skip the correction and clamp the token count; the
# update then trains on nothing instead of dying. Third guard: the hybrid_engine=False branch of the v1
# separate_async trainer calls logger.info(..., flush=True), a TypeError at init. All fail loudly if a line moved.
python3 - <<'PY'
import pathlib
f = pathlib.Path("/workspace/uni-agent/verl/verl/trainer/ppo/rollout_corr_helper.py"); s = f.read_text()
old = ('    # Validate input masks\n    if not response_mask.any():\n'
       '        raise ValueError("response_mask must contain at least one valid token (1).")\n')
new = ('    # Validate input masks\n    if not response_mask.any():\n'
       '        print("WARNING: rollout correction skipped: no valid response token in this batch (all samples loss-masked)")\n'
       '        return None, response_mask, {"rollout_corr/fully_masked_batch": 1.0}\n')
if new in s: print("verl rollout correction already tolerates fully-masked batches")
elif s.count(old) == 1: f.write_text(s.replace(old, new)); print("patched verl rollout correction to skip fully-masked batches")
else: raise SystemExit("verl rollout correction changed; refusing to patch " + str(f))
f = pathlib.Path("/workspace/uni-agent/verl/verl/workers/engine/megatron/transformer_impl.py"); s = f.read_text()
old = "        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())\n"
new = "        tu.assign_non_tensor(data, batch_num_tokens=max(batch_num_tokens.item(), 1))  # fully-masked batch: loss 0, not 0/0\n"
if new in s: print("verl Megatron loss normalization already clamps the token count")
elif s.count(old) == 1: f.write_text(s.replace(old, new)); print("patched verl Megatron loss normalization for fully-masked batches")
else: raise SystemExit("verl Megatron engine changed; refusing to patch " + str(f))
f = pathlib.Path("/workspace/uni-agent/verl/verl/trainer/ppo/v1/trainer_separate_async.py"); s = f.read_text()
old = 'standalone replicas only",\n                flush=True,\n            )'
new = 'standalone replicas only",\n            )'
if new in s and old not in s: print("verl separate_async logger call already fixed")
elif s.count(old) == 1: f.write_text(s.replace(old, new)); print("patched verl separate_async logger call (flush kwarg) on the hybrid_engine=False path")
else: raise SystemExit("verl separate_async trainer changed; refusing to patch " + str(f))
PY
python3 - <<'PY'
import importlib
for m in ("verl.trainer.main_ppo", "fla", "megatron.bridge", "cupy", "swebench", "ate_env", "uni_agent_substrate.provider"):
    importlib.import_module(m)
from uni_agent.sandbox.registry import get_sandbox_cls; from uni_agent.tasks.registry import get_task_cls
import verl, vllm, transformers
print("ok:", "verl", verl.__version__, "vllm", vllm.__version__, "transformers", transformers.__version__,
      get_sandbox_cls("ate_env").__name__, get_task_cls("r2e_gym").__name__)
PY

# Head only, before the (long) model pull: prove ate-env-api, the run's actor template and image resolution work,
# with the template and a digest-pinned image the run will actually use.
if [ "$ROLE" = "head" ] && [ "${DRY_RUN:-0}" != "1" ]; then
  echo "=== Phase 1b (head): ate-env-api probe on template $ATE_TEMPLATE ==="
  PROBE_IMAGE=$(gunzip -c /workspace/cfg/images.json.gz | python3 -c 'import json, sys; print(next(iter(json.load(sys.stdin).values())))')
  # Same patience as the provider: a wake that misses the server's budget (pool busy) is retried on the same
  # environment up to SANDBOX_START_WAIT_S; anything else fails the launch before the model pull.
  python3 - "$ATE_ENV_ENDPOINT" "$ATE_TEMPLATE" "$PROBE_IMAGE" "$ACTOR_PREFIX" "$SANDBOX_START_WAIT_S" <<'PY'
import asyncio, sys, time, uuid
from ate_env import Client
from ate_env.errors import EnvError
endpoint, template, image, prefix, wait_s = sys.argv[1:6]
async def probe():
    c = Client(endpoint)
    env = await c.create(f"{prefix}-probe-{uuid.uuid4().hex[:8]}", template_name=template, rootfs_image=image)
    t0 = time.monotonic(); attempt = 0
    try:
        while True:
            attempt += 1
            try:
                r = await env.shell("echo 42", timeout=600)
                break
            except EnvError as exc:
                if "did not answer within the wake budget" not in str(exc) or time.monotonic() - t0 > float(wait_s):
                    raise
                print(f"probe: sandbox pool busy, waiting ({time.monotonic() - t0:.0f}s so far, attempt {attempt}): {str(exc)[:160]}")
                await asyncio.sleep(10)
        assert r.exit_code == 0 and "42" in r.stdout, (r.exit_code, r.stdout, r.stderr)
    finally:
        for _ in range(12):   # delete is Aborted while the actor is still mid-resume
            try:
                await env.delete(); break
            except EnvError as exc:
                if "another operation is in progress" not in str(exc): break
                await asyncio.sleep(10)
        await c.close()
    print(f"ate-env-api probe ok in {time.monotonic() - t0:.0f}s: {endpoint}, template {template}, image {image.rsplit('/', 1)[-1][:48]}")
asyncio.run(probe())
PY
fi

echo "=== Phase 2: model $MODEL_ID ==="
MODEL_PATH=/workspace/models/${MODEL_ID##*/}
DRY_RUN="${DRY_RUN:-0}" python3 - <<PY
import os, shutil, time
from huggingface_hub import HfApi, snapshot_download
if os.environ["DRY_RUN"] == "1":   # config + tokenizer only: enough to compose and validate the verl config
    snapshot_download(repo_id="$MODEL_ID", local_dir="$MODEL_PATH", allow_patterns=["*.json", "*.txt", "*.py", "merges.txt"])
    print("dry run: downloaded config/tokenizer only"); raise SystemExit
need = sum((s.size or 0) for s in HfApi().model_info("$MODEL_ID", files_metadata=True).siblings)
free = shutil.disk_usage("/workspace").free
print(f"checkpoint {need/1e9:.0f} GB, free under /workspace {free/1e9:.0f} GB", flush=True)
assert free > need * 1.15 + 50e9, "not enough ephemeral storage for the checkpoint: raise the pod's ephemeral-storage request / give the node pool local SSD"
t = time.time(); snapshot_download(repo_id="$MODEL_ID", local_dir="$MODEL_PATH", max_workers=16)
print(f"model ready: $MODEL_PATH ({(time.time()-t)/60:.1f} min, {need/1e9/max(time.time()-t,1):.2f} GB/s)", flush=True)
PY
# Online FP8 rollout: verl quantizes every synced weight to block-FP8 for vLLM; the modules the official
# Qwen3.5 FP8 checkpoints keep in bf16 (MoE routers, GDN conv/low-rank projections, embeddings, lm_head,
# the vision tower whose 1152-wide layers do not tile by 128) must stay unquantized here too.
FP8_OVERRIDES=()
if [ -n "$ROLLOUT_QUANT" ]; then
  python3 - "$MODEL_PATH/config.json" > /workspace/fp8_overrides.txt <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1])); text, vis = cfg.get("text_config", cfg), cfg.get("vision_config") or {}
keep = ["lm_head", "model.language_model.embed_tokens"]
for i, kind in enumerate(text["layer_types"]):
    p = f"model.language_model.layers.{i}"
    if kind == "linear_attention":
        keep += [f"{p}.linear_attn.conv1d", f"{p}.linear_attn.in_proj_a", f"{p}.linear_attn.in_proj_b"]
    keep += [f"{p}.mlp.gate", f"{p}.mlp.shared_expert_gate"]
for i in range(vis.get("depth", 0)):
    p = f"model.visual.blocks.{i}"
    keep += [f"{p}.attn.proj", f"{p}.attn.qkv", f"{p}.mlp.linear_fc1", f"{p}.mlp.linear_fc2"]
if vis:
    keep += ["model.visual.merger.linear_fc1", "model.visual.merger.linear_fc2", "model.visual.patch_embed.proj", "model.visual.pos_embed"]
for i in range(text.get("mtp_num_hidden_layers", 0) or 0):
    keep += ["mtp.fc", f"mtp.layers.{i}.mlp.gate", f"mtp.layers.{i}.mlp.shared_expert_gate"]
q = "+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.quantization_config."
print(q + "quant_method=fp8"); print(q + "activation_scheme=dynamic"); print(q + "fmt=e4m3"); print(q + "weight_block_size=[128,128]")
print(q + "modules_to_not_convert=[" + ",".join(f"'{m}'" for m in keep) + "]")
print(f"fp8: {len(keep)} modules kept in bf16", file=sys.stderr)
PY
  mapfile -t FP8_OVERRIDES < /workspace/fp8_overrides.txt
fi

# Pipeline split: Megatron needs equal middle stages; when num_hidden_layers is not divisible by TRAIN_PP (or
# PP_EDGE_LAYERS is forced) give the first and last stage fewer layers (the DeepSeek-671B recipe's trick).
PP_SPLIT=()
if [ "$TRAIN_PP" -gt 1 ]; then
  EDGE=$(python3 - <<PY
import json
cfg = json.load(open("$MODEL_PATH/config.json")); cfg = cfg.get("text_config", cfg)
L, pp, forced = cfg["num_hidden_layers"], $TRAIN_PP, "$PP_EDGE_LAYERS"
def ok(edge):
    return (pp == 2 and 2 * edge == L) or (pp > 2 and edge >= 1 and (L - 2 * edge) % (pp - 2) == 0 and (L - 2 * edge) // (pp - 2) >= 1)
if forced:
    edge = int(forced); assert ok(edge), f"PP_EDGE_LAYERS={edge} does not split {L} layers over PP={pp}"
elif L % pp == 0:
    edge = 0
else:
    edge = next((e for e in range(L // pp, 0, -1) if ok(e)), None); assert edge, f"no uneven split of {L} layers over PP={pp}"
print(edge)
PY
)
  if [ "$EDGE" != "0" ]; then
    echo "pipeline split: first/last stage $EDGE layers"
    PP_SPLIT=(+actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage="$EDGE"
              +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage="$EDGE"
              +actor_rollout_ref.actor.megatron.override_transformer_config.account_for_embedding_in_pipeline_split=False
              +actor_rollout_ref.actor.megatron.override_transformer_config.account_for_loss_in_pipeline_split=False)
  fi
fi

# Every pod: the uni-agent task runners are Ray tasks and run on any node, so every node needs the
# task config and the image digest map at the same paths.
gunzip -c /workspace/cfg/images.json.gz > /workspace/agent/images.json
export ATE_ENV_IMAGE_DIGESTS=/workspace/agent/images.json
sed -e "s|\${ATE_ENV_ENDPOINT}|$ATE_ENV_ENDPOINT|g" -e "s|\${ATE_TEMPLATE}|$ATE_TEMPLATE|g" -e "s|\${AGENT_MAX_STEPS}|$AGENT_MAX_STEPS|g" -e "s|\${ACTOR_PREFIX}|$ACTOR_PREFIX|g" \
    -e "s|\${SANDBOX_START_WAIT_S}|$SANDBOX_START_WAIT_S|g" \
    /workspace/cfg/task_config.yaml > /workspace/agent/task_config.yaml

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "=== DRY_RUN: compose and validate the verl config (no datasets, no Ray, no GPUs) ==="
  mkdir -p /workspace/data; touch /workspace/data/train.parquet /workspace/data/val.parquet
  ROLE=head; SKIP_RUNTIME=1
fi
if [ "$ROLE" = "worker" ]; then
  echo "=== Phase 3 (worker): join Ray head $RAY_HEAD_HOST:$RAY_PORT ==="
  for i in $(seq 1 360); do getent hosts "$RAY_HEAD_HOST" >/dev/null && break || sleep 10; done
  ray start --address="$RAY_HEAD_HOST:$RAY_PORT" --num-gpus=8 --num-cpus="$RAY_CPUS" --block || true   # returns when the head stops Ray
  echo "=== worker done ==="; exit 0
fi

if [ "${SKIP_RUNTIME:-0}" != "1" ]; then
echo "=== Phase 3 (head): datasets ==="
python3 -m uni_agent_substrate.datasets --source "$TRAIN_SOURCE" --split "$TRAIN_SPLIT" ${TRAIN_DATASET:+--dataset "$TRAIN_DATASET"} \
    --shuffle-seed "$TRAIN_SEED" --max-instances "$TRAIN_INSTANCES" --image-digests /workspace/agent/images.json --out /workspace/data/train.parquet
python3 -m uni_agent_substrate.datasets --source swe_bench --max-instances "$VAL_TASKS" --exclude-file /workspace/cfg/offline-exclusions.txt \
    --image-digests /workspace/agent/images.json --out /workspace/data/val.parquet

echo "=== Phase 4: Ray head + wait for $NNODES GPU nodes ==="
ray start --head --port="$RAY_PORT" --num-gpus=8 --num-cpus="$RAY_CPUS" --disable-usage-stats
export RAY_ADDRESS="127.0.0.1:$RAY_PORT"
NNODES="$NNODES" python3 - <<'WAIT'
import os, time, ray
ray.init(address=os.environ["RAY_ADDRESS"], log_to_driver=False)
need, deadline = int(os.environ["NNODES"]), time.time() + int(os.environ["NODE_WAIT_S"])
while True:
    alive = [n for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0) > 0]
    print(f"ray nodes with GPUs: {len(alive)}/{need}", [(n["NodeManagerAddress"], n["Resources"].get("GPU")) for n in alive], flush=True)
    if len(alive) >= need:
        break
    assert time.time() < deadline, f"only {len(alive)} GPU nodes joined within NODE_WAIT_S"
    time.sleep(15)
# verl's driver task and the uni-agent runners may land on any node: give every node the dataset files.
import pathlib
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
files = {str(f): f.read_bytes() for f in pathlib.Path("/workspace/data").glob("*.parquet")}
@ray.remote(num_cpus=0)
def put(blobs):
    for path, data in blobs.items():
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True); pathlib.Path(path).write_bytes(data)
    return True
ray.get([put.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=n["NodeID"], soft=False)).remote(files) for n in alive])
print(f"dataset files replicated to {len(alive)} nodes: {sorted(files)}", flush=True)
WAIT
fi

echo "=== Phase 5: verl separate_async ==="
TOKENS_PER_GPU=$((MAX_MODEL_LEN / TRAIN_CP))
: "${LR_WARMUP_STEPS:=10}"; [ "$LR_WARMUP_STEPS" -lt "$TOTAL_STEPS" ] || LR_WARMUP_STEPS=$((TOTAL_STEPS - 1))   # Megatron: warmup < decay steps
HYBRID=()
if [ "$ROLLOUT_HYBRID_FLAGS" = "1" ]; then
  HYBRID=(+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL_DECODE_ONLY
          +actor_rollout_ref.rollout.engine_kwargs.vllm.mamba_cache_mode=align)
fi
QUANT=(); [ -n "$ROLLOUT_QUANT" ] && QUANT=(actor_rollout_ref.rollout.quantization="$ROLLOUT_QUANT")
# Multi-NIC / dual-stack nodes: pin the interface NCCL and Gloo advertise (otherwise Gloo may pick an address other
# nodes cannot route, e.g. an IPv6 one, and Megatron's init fails with "connectFullMesh ... Network is unreachable").
NCCL_IF=()
# quoted: a comma-separated interface list (e.g. eth0,eth1) would otherwise be parsed by Hydra as a sweep
[ -n "${NCCL_SOCKET_IFNAME:-}" ] && NCCL_IF+=("+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_IFNAME='$NCCL_SOCKET_IFNAME'")
[ -n "${GLOO_SOCKET_IFNAME:-}" ] && NCCL_IF+=("+ray_kwargs.ray_init.runtime_env.env_vars.GLOO_SOCKET_IFNAME='$GLOO_SOCKET_IFNAME'")
cd /workspace/results   # not the uni-agent checkout: its docker/ folder would shadow the docker package for Ray workers
OVERRIDES=(
  model_engine=megatron
  '+ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_OFFLINE="1"'
  "+ray_kwargs.ray_init.runtime_env.env_vars.SANDBOX_STARTUP_CONCURRENCY=\"$SANDBOX_STARTUP_CONCURRENCY\""   # quoted: Ray env vars must be strings
  "+ray_kwargs.ray_init.runtime_env.env_vars.SANDBOX_STARTUP_TIMEOUT=\"$((SANDBOX_START_WAIT_S + 600))\""     # uni-agent's per-start cap must cover the capacity wait
  "+ray_kwargs.ray_init.runtime_env.env_vars.ATE_ENV_IMAGE_DIGESTS=$ATE_ENV_IMAGE_DIGESTS"
  '+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_DISABLE_COMPILE_CACHE="1"'
  '+ray_kwargs.ray_init.runtime_env.env_vars.CUDA_DEVICE_MAX_CONNECTIONS="1"'
  "${NCCL_IF[@]}"
  trainer.use_v1=True
  trainer.v1.trainer_mode=separate_async
  actor_rollout_ref.hybrid_engine=False   # at this pin separate_async otherwise ALSO starts vLLM replicas on the trainer nodes (host RAM for their sleeping weights + mode switching)
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=$ROLLOUT_DISABLE_CUSTOM_ALLREDUCE"
  trainer.v1.separate_async.num_warmup_batches=1
  trainer.v1.separate_async.parameter_sync_step="$SYNC_STEP"
  trainer.v1.sampler.max_off_policy_threshold=16
  transfer_queue.enable=True
  actor_rollout_ref.nccl_timeout=9600
  actor_rollout_ref.model.path="$MODEL_PATH"
  actor_rollout_ref.model.trust_remote_code=True
  actor_rollout_ref.model.use_remove_padding=False
  actor_rollout_ref.model.mtp.enable=False
  +actor_rollout_ref.model.override_config.model_config.max_position_embeddings=$MAX_MODEL_LEN
  data.train_files=/workspace/data/train.parquet
  data.val_files=/workspace/data/val.parquet
  data.prompt_key=prompt
  data.return_raw_chat=True
  data.truncation=left
  data.filter_overlong_prompts=True
  data.trust_remote_code=True
  data.dataloader_num_workers=0
  data.max_prompt_length=$PROMPT_LENGTH
  data.max_response_length=$RESPONSE_LENGTH
  data.train_batch_size="$TRAIN_BATCH"
  data.val_batch_size="$VAL_TASKS"
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.dtype=bfloat16
  actor_rollout_ref.rollout.nnodes="$NNODES_ROLLOUT"
  actor_rollout_ref.rollout.n_gpus_per_node=8
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP"
  actor_rollout_ref.rollout.data_parallel_size="$ROLLOUT_DP"
  actor_rollout_ref.rollout.expert_parallel_size="$ROLLOUT_EP"
  "${QUANT[@]}"
  "${FP8_OVERRIDES[@]}"
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEM_UTIL"
  actor_rollout_ref.rollout.n="$ROLLOUT_N"
  actor_rollout_ref.rollout.prompt_length=$PROMPT_LENGTH
  actor_rollout_ref.rollout.response_length=$RESPONSE_LENGTH
  actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN
  actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_BATCHED_TOKENS"
  actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS"
  actor_rollout_ref.rollout.enable_chunked_prefill=True
  actor_rollout_ref.rollout.calculate_log_probs=True
  actor_rollout_ref.rollout.enforce_eager=False
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.val_kwargs.n=1
  actor_rollout_ref.rollout.val_kwargs.do_sample=True
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95
  actor_rollout_ref.rollout.checkpoint_engine.backend=nccl
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$TOKENS_PER_GPU
  "${HYBRID[@]}"
  actor_rollout_ref.rollout.multi_turn.enable=True
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
  ++actor_rollout_ref.rollout.multi_turn.format=qwen3_coder
  actor_rollout_ref.rollout.agent.num_workers="$AGENT_WORKERS"
  ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter
  ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count="$GATEWAYS"
  ++actor_rollout_ref.rollout.custom.agent_framework.log_dir=/workspace/results/agent-logs
  ++actor_rollout_ref.rollout.custom.agent_framework.allowed_request_sampling_param_keys="[temperature,top_p,top_k]"
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=ray_task
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions="$SESSIONS"
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.session_timeout_seconds="$SESSION_TIMEOUT_S"
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.trajectory_selection=longest
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path=/workspace/agent/task_config.yaml
  "++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name=${MODEL_ID##*/}"
  ++actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode="$MASK_UNFINISHED_EPISODE"
  trainer.nnodes="$NNODES_TRAIN"
  trainer.n_gpus_per_node=8
  actor_rollout_ref.actor.megatron.use_mbridge=True
  actor_rollout_ref.actor.megatron.vanilla_mbridge=True
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size="$TRAIN_TP"
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="$TRAIN_PP"
  actor_rollout_ref.actor.megatron.context_parallel_size="$TRAIN_CP"
  actor_rollout_ref.actor.megatron.expert_model_parallel_size="$TRAIN_EP"
  actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1
  actor_rollout_ref.actor.megatron.use_remove_padding=False
  actor_rollout_ref.actor.megatron.param_offload="$PARAM_OFFLOAD"
  actor_rollout_ref.actor.megatron.optimizer_offload=True
  actor_rollout_ref.actor.megatron.dtype=bfloat16
  ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
  +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=False
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
  "${PP_SPLIT[@]}"
  actor_rollout_ref.actor.use_dynamic_bsz=False
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$TOKENS_PER_GPU
  actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH"
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.optim.lr_warmup_steps="$LR_WARMUP_STEPS"
  actor_rollout_ref.actor.optim.lr_decay_style=constant
  actor_rollout_ref.actor.optim.weight_decay=0.1
  actor_rollout_ref.actor.optim.clip_grad=1.0
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1.0
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
  +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
  +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
  actor_rollout_ref.actor.clip_ratio_low=0.2
  actor_rollout_ref.actor.clip_ratio_high=0.28
  actor_rollout_ref.actor.clip_ratio_c=10.0
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.loss_agg_mode=token-mean
  actor_rollout_ref.actor.checkpoint.strict=False
  algorithm.rollout_correction.rollout_is=token
  algorithm.rollout_correction.rollout_is_threshold=2.0
  actor_rollout_ref.ref.megatron.param_offload=True
  actor_rollout_ref.ref.megatron.tensor_model_parallel_size="$TRAIN_TP"
  actor_rollout_ref.ref.megatron.pipeline_model_parallel_size="$TRAIN_PP"
  actor_rollout_ref.ref.megatron.context_parallel_size="$TRAIN_CP"
  actor_rollout_ref.ref.megatron.expert_model_parallel_size="$TRAIN_EP"
  actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=1
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$TOKENS_PER_GPU
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  reward.reward_manager.name=dapo
  reward.custom_reward_function.path=pkg://uni_agent.framework.task_runner
  reward.custom_reward_function.name=score_from_runner_result
  trainer.val_before_train=False
  trainer.test_freq="$TOTAL_STEPS"
  trainer.total_epochs=1
  trainer.total_training_steps="$TOTAL_STEPS"
  trainer.save_freq=-1
  trainer.resume_mode=disable
  trainer.rollout_data_dir=/workspace/results/rollouts
  trainer.validation_data_dir=/workspace/results/val-rollouts
  trainer.logger='["console"]'
  trainer.log_val_generations=2
  trainer.project_name=verl-agentic-swe
  "trainer.experiment_name=$(echo "${MODEL_ID##*/}" | tr 'A-Z' 'a-z')-react-substrate"
)
if [ "${DRY_RUN:-0}" = "1" ]; then
  printf '%s\n' "${OVERRIDES[@]}"
  # Hydra composition + every config dataclass's post-init checks (parallelism, rollout, megatron) without GPUs
  python3 /workspace/uni-agent/verl/scripts/print_cfg.py --config-name=ppo_megatron_trainer "${OVERRIDES[@]}" > /workspace/results/resolved_cfg.txt
  grep -nE "trainer_mode|quantization|pipeline_model_parallel_size|num_layers_in_(first|last)|nnodes" /workspace/results/resolved_cfg.txt | head -20
  echo "=== DRY_RUN OK ==="; exit 0
fi
python3 -m verl.trainer.main_ppo --config-name=ppo_megatron_trainer "${OVERRIDES[@]}" \
  2>&1 | tee /workspace/results/train.log
RC=${PIPESTATUS[0]}
ray stop --force || true          # releases the worker pods
echo "=== DONE rc=$RC ==="
