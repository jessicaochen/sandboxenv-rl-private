# Qwen3.5-397B-A17B agentic GRPO on SWE tasks: verl + uni-agent + Substrate

A uni-agent ReAct agent fixes real repository issues inside Agent Substrate environments. Hidden tests
run in the same environment and decide the reward. GRPO on full weights with verl v1 `separate_async`:
Megatron trainer on `NNODES_TRAIN` nodes, FP8 vLLM replicas on `NNODES_ROLLOUT` other nodes, the agent's
tool calls and the grading in Substrate through `ate-env-api`.

Pins: uni-agent `174b040` (its verl submodule `3efe38c7`), image `verlai/verl:vllm024.dev2`,
`swebench==4.1.0`. The `ate_env` client ([`clients/python`](../../clients/python)) and the provider, task
and dataset tooling ([`integrations/uni-agent`](../../integrations/uni-agent)) ship from this checkout in
the payload ConfigMap; the pods need no access to this repository.

Same files at every scale; only `profiles/<name>.env` changes. The two 35B profiles are sized so that eight of
them (`JOB_SUFFIX=a..h`) fill the same 16 GPU nodes as one 397B profile and generate as many trajectories per
step: eight `35b-easy` match `397b-easy`, eight `35b-v1` match `397b-v1`.

| Profile | Nodes | Model | Trainer | Rollout | Step | Dataset, budget |
|---|---|---|---|---|---|---|
| `35b-easy` | 2 | Qwen3.5-35B-A3B | 1 node, TP2 PP2 EP4 | 1 node, 4 x TP2 FP8 | 16 x 8 = 128 | R2E-Gym Easy, 8K / 32K, 100 steps |
| `35b-v1` | 2 | Qwen3.5-35B-A3B | same | same | 128 x 16 = 2,048 | R2E-Gym-V1, 8K / 32K, 100 steps |
| `397b-smoke` | 16 | Qwen3.5-397B-A17B | 8 nodes, TP2 PP8 EP8 | 8 nodes, 8 x TP8 FP8 | 16 x 8 = 128 | R2E-Gym-V1, 8K / 32K, 100 steps |
| `397b-easy` | 16 | Qwen3.5-397B-A17B | same | same | 64 x 16 = 1,024 | R2E-Gym Easy, 8K / 32K, 100 steps |
| `397b-v1` | 16 | Qwen3.5-397B-A17B | same | same | 1,024 x 16 = 16,384 | R2E-Gym-V1, 8K / 64K, 150 steps |

Every profile runs 2 steps plus validation by default (`TOTAL_STEPS`); raise it and `TRAIN_INSTANCES` for a
training run. `SESSIONS` is `min(step, replicas x ROLLOUT_MAX_NUM_SEQS)`: the trajectories a replica cannot decode
at once wait in uni-agent, holding no sandbox, instead of queueing inside vLLM. `35b-v1` and `397b-v1` run half
and a quarter of their step at a time; the step takes the same time either way.

## Prerequisites

- Agent Substrate and `ate-env-api` in namespace `ate-env` at the commit this repo's `go.mod` pins;
  `kubectl-ate` and `ate-env` from that commit. The profile's `ATE_TEMPLATE` must exist (step 1 creates
  both); the head probes it before the model pull. If `ate-env-api` requires a caller token, export
  `SUBSTRATE_ENV_AUTH_TOKEN` before `deploy.sh up`.
- GPU nodes with 8 GPUs each: 2 for the 35B profiles, 16 for the 397B. Sized for 80 GB GPUs; the trainer's offloaded
  params, optimizer and Decoupled-PPO weight copies need most of the host RAM, so the pods carry no
  memory limit. Taint `nvidia.com/gpu` is tolerated, others go in `GPU_TOLERATIONS`.
- `EPH_REQUEST` of ephemeral storage per GPU pod for the checkpoint (1 TB for the 397B): local SSD or a
  large boot disk. The run script checks free space before the pull.
- A CPU pool for Substrate workers, labeled and tainted `ate.dev/sandboxClass=gvisor` (Substrate tolerates
  that taint itself). Node disk: distinct images the node will boot x 0.6 GB, plus 50 GB.
- Task images reachable as digest-pinned references: an Artifact Registry remote repo proxying Docker Hub,
  or a mirror.
- On the prep machine: `crane`, Python 3.11+, `pip install -e integrations/uni-agent[datasets]`. GPU pods
  need egress to GitHub (uni-agent), PyPI and Hugging Face.

## Run

```bash
export KUBE_CONTEXT=<context> KUBECTL_ATE=<path to kubectl-ate>

# 1. Substrate workers + templates (once). replicas = ceil((SESSIONS + VAL_TASKS) * 1.15 / environments per node)
kubectl -n ate-env patch workerpool default-template-workerpool --type=merge -p '{"spec":{"replicas":20,"template":{
  "nodeSelector":{"ate.dev/sandboxClass":"gvisor"},
  "resources":{"requests":{"cpu":"12","memory":"110Gi"},"limits":{"cpu":"12","memory":"110Gi"}}}}}'
ate-env manifest template --template swe-1c4g --cpu 1 --memory 4Gi --workerpool default-template-workerpool \
    --atespace ate-env | kubectl-ate create actor-template -f -      # ATE_TEMPLATE; add --snapshots-bucket / --guest-image as in the Quickstart

# 2. data slices + digest-pinned images (once per slice; match TRAIN_SOURCE / TRAIN_SEED / TRAIN_INSTANCES in the profile)
python3 -m uni_agent_substrate.datasets --source r2e_gym_easy --max-instances 128 --images-out images.txt --out /tmp/train.parquet   # *-easy
python3 -m uni_agent_substrate.datasets --source r2e_gym --dataset R2E-Gym/R2E-Gym-V1 --shuffle-seed 0 \
    --max-instances 4096 --images-out images.txt --out /tmp/train.parquet                                                  # *-v1
python3 -m uni_agent_substrate.datasets --source swe_bench --max-instances 20 \
    --exclude-file ../../benchmarking/swebench/offline-exclusions.txt --images-out images-val.txt --out /tmp/val.parquet
MODE=hubapi ../../integrations/uni-agent/scripts/resolve_images.sh images.txt <region>-docker.pkg.dev/<project>/<dockerhub-remote-repo> images.json
../../integrations/uni-agent/scripts/resolve_images.sh images-val.txt <region>-docker.pkg.dev/<project>/<dockerhub-remote-repo> images.json 6

# 3. 35B on 2 nodes, 397B smoke on 16, then the 397B shape you want
PROFILE=35b-easy ./deploy.sh up && ./deploy.sh logs      # ... ray nodes with GPUs: 2/2 ... step:1 ... step:2 ... === DONE rc=0 ===
./deploy.sh down
PROFILE=397b-smoke ./deploy.sh up && ./deploy.sh logs    # 16/16 nodes
./deploy.sh down
./deploy.sh up && ./deploy.sh logs                       # 397b-v1 (default), or PROFILE=397b-easy
./deploy.sh down
```

- `images.json` holds your registry paths; git-ignored, read from `IMAGES_JSON`, shipped gzipped in the
  ConfigMap (thousands of entries fit).
- Knobs are the keys of `profiles/<name>.env`, documented inline; `PROFILE_FILE=<path>` points at a copy kept
  elsewhere (your own node counts or knobs). `./deploy.sh render` prints the manifests
  without a cluster; `DRY_RUN=1` in a profile makes the head compose and validate the verl config and exit
  (a CPU pod is enough).
- `GPU_POD_ENV=K=V,K2=V2` adds environment variables to the GPU pods: NCCL and Gloo settings for your
  network (`NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` on multi-NIC or dual-stack nodes, plugin toggles);
  both interface pins are also forwarded to every Ray worker.
- `JOB_SUFFIX=a ./deploy.sh up` (same value for `logs` / `down`) runs several instances in one namespace, e.g.
  eight `35b-easy` on 16 nodes:
  Jobs, Service, ConfigMaps, labels and environment id prefix are all suffixed, so instances neither
  collide nor purge each other's environments.

## Why these settings

- **Trainer fits 80 GB GPUs** by verl's DeepSeek-671B guide: params and Adam state on the CPU between
  steps, PP across nodes with an uneven first/last stage, TP/EP inside a node (`alltoall`, no RDMA),
  `moe_router_dtype=fp32`, `apply_rope_fusion=False`. Fewer pipeline stages put more layers per rank; raise
  `TRAIN_TP` then. Never set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` pod-wide: the checkpoint-engine
  to vLLM weight handoff then needs `pidfd_getfd` and fails with "Operation not permitted".
- **Disaggregated needs `hybrid_engine=False`.** At this pin `separate_async` otherwise also starts vLLM
  replicas on the trainer nodes; their sleeping weights sit in the trainer's host RAM. The install step
  patches that path's logging call (a `flush=True` TypeError).
- **One replica per node** via online FP8: bf16 weights (807 GB) do not fit a node and verl's vLLM cannot
  span nodes. Megatron stays bf16, each synced bucket is block-quantized into vLLM, routers stay
  unquantized, `rollout_is=token` corrects the mismatch; the 35B profiles use the same path. bf16
  fallback: `ROLLOUT_QUANT=` with `ROLLOUT_DP=2 ROLLOUT_EP=16`. Replicas use NCCL all-reduce
  (`ROLLOUT_DISABLE_CUSTOM_ALLREDUCE`); vLLM's IPC kernel can fail at start on nodes still draining a
  previous run.
- **Qwen3.5 specifics** (verl's `run_qwen3_5_397b_megatron.sh`): GDN has no packed sequences in Megatron,
  so BSHD without dynamic batching; no context parallelism (`TRAIN_CP=1`); vanilla mbridge; full
  recompute; vLLM `cudagraph_mode=FULL_DECODE_ONLY` and `mamba_cache_mode=align`; MTP off. Response
  length is bounded by one GPU's activations: longer contexts need more pipeline stages, not CP.
- **Episode budget** (uni-agent's 235B SWE example): 64K responses, 150 steps, 128K one knob away; clip
  0.2/0.28, lr 1e-6, weight decay 0.1.
- **Unfinished episodes** (`MASK_UNFINISHED_EPISODE=True`): an episode that hits `AGENT_MAX_STEPS` or
  `RESPONSE_LENGTH` keeps its reward in the GRPO group but trains on no tokens, so the budget must be one
  most episodes finish within. A batch with no trainable token is tolerated (verl is patched at install to
  skip rollout correction and train on nothing instead of raising `response_mask must contain at least
  one valid token`); when the few finished episodes sit in uniform-reward groups the step logs `grad_norm`
  0. Set `False` to train on truncated episodes, or raise the budget.

## Operating at scale

For steps of thousands of trajectories. Profile values and Substrate settings only.

- **`SESSIONS` = replicas x `ROLLOUT_MAX_NUM_SEQS`.** What vLLM cannot decode at once should wait in uni-agent,
  where a waiting episode holds no sandbox and can go to whichever replica frees first, rather than inside
  vLLM. The sandbox pool then needs about `SESSIONS` environments, not a whole step. Sessions beyond the
  pool's capacity still wait up to `SANDBOX_START_WAIT_S` instead of failing. Each in-flight session is a
  Ray task and worker process: `RAY_CPUS` >= `SESSIONS / GPU nodes`, `SESSION_TIMEOUT_S` covers wait plus episode.
- **The startup wave is gated job-wide.** The provider applies `SANDBOX_STARTUP_CONCURRENCY` through one
  Ray actor per job (uni-agent's own semaphore is per process), keeps waiting on the same environment when
  a wake is merely slow, retries deletes Substrate rejects as "another operation is in progress", and
  reclaims environments whose session never got ready or died. Size the gate so `SESSIONS / gate` times the
  start latency (20 to 45 s warm) stays under `SANDBOX_START_WAIT_S`.
- **Step time is decode-bound.** Generation time is trajectories x tokens over the replicas' aggregate decode
  rate, so more trajectories per step than `SESSIONS` means more waves, not slower waves; add rollout nodes
  to shorten it. The update scales the same way on the trainer nodes and at these batches takes about as
  long as generation; plan the window around both.
- **Sandbox disk is the limit, not memory.** Substrate caches every image a node has booted (~0.6 GB each);
  a full disk evicts the worker and crashes every environment it hosts. A 2 TB disk holds ~3,000 images.
- **Booking 1 vCPU / 4 GiB per environment on nodes with >= 8 GB per vCPU.** The worker's memory limit is
  the real cap; agent episodes peak well above the booking, and an OOM-killed worker takes every hosted
  environment with it. Keep 2 vCPU and 1 GiB per node for atelet.
- **Drain before nodes go away.** Environments whose node disappears mid-delete stay `DELETING` forever;
  run `deploy.sh down` before a sandbox pool scales down or recreates nodes.
- **Control plane for thousands of environments.** ate-api-server 4 pods with
  `ATE_API_POSTGRES_POOL_MAX_CONNS=64`, Postgres `max_connections >= pods x 69 + 50` on 8 vCPU / 16 GiB
  SSD (`shared_buffers=4GB`, `max_wal_size=16GB`; or installer `clusterSize: size10`), atenet-router 1 pod
  per ~1,000 concurrent requests, ate-env-api `--api-replicas` 4 to 8. Re-apply after every deploy.
- **Docker Hub limits.** `MODE=hubapi` resolves digests through the tags API (no pull quota); SWE-bench's
  `swebench/*` repos are not on it, resolve those with crane at parallelism 6. Pulls stay rate-limited:
  give the remote repo upstream credentials before thousands of images. `MODE=arapi` and `MIRROR_NAME_SEP`
  resolve against an existing mirror.

## Files

| File | Purpose |
|---|---|
| `deploy.sh` | `up` / `render` / `logs` / `down`; renders the manifest from the profile, packs the ConfigMaps, purges leftover environments |
| `profiles/*.env` | `35b-easy`, `35b-v1`, `397b-smoke`, `397b-easy`, `397b-v1`: the shapes; every key becomes pod env |
| `job-fully-async.yaml` | Ray head Job + indexed worker Job + headless Service, pod anti-affinity, 8 GPUs each |
| `run_fully_async.sh` | every pod: topology checks, pinned install with guarded verl patches, model download, pipeline split; head: datasets, probe, Ray, verl `separate_async` |
| `task_config.yaml` | uni-agent tasks `r2e_gym` (train) and `swe_bench` (validation) on the `ate_env` provider; ReAct tools and SWE prompt |

## What it covers

| Property | How |
|---|---|
| Agentic | uni-agent ReAct agent, `AGENT_MAX_STEPS` tool steps, tools `str_replace_editor` / `stateful_shell` / `submit`, parser `qwen3_coder` |
| GRPO, full weights | `adv_estimator=grpo`, n=8, Megatron actor via mbridge (TP2 PP8 EP8 at scale, CPU-offloaded params and optimizer), no LoRA, bf16 training |
| Async | `trainer.v1.trainer_mode=separate_async`, `parameter_sync_step=2`, off-policy threshold 16, token-level importance weights |
| Disaggregated | two Jobs with required anti-affinity; `trainer.nnodes` and `rollout.nnodes` never share a machine; `hybrid_engine=False`; weights pushed over NCCL, FP8-quantized on arrival |
| Substrate via ate-env-api | provider `ate_env`: one environment per episode from the instance's digest-pinned image on `ATE_TEMPLATE` |
| In-rollout tool execution | shell and editor calls run in that environment (`Env.start_process`, native file RPCs, no egress) |
| Post-hoc reward executor | `r2e_gym`: hidden tests archived before the agent, restored after, `run_tests.sh` graded against `expected_output_json`; `swe_bench`: uni-agent's harness. Check: `run_oracle_solution: true` must score 1.0 |
| Swappable dataset | `uni_agent_substrate.datasets --source {r2e_gym_easy,r2e_gym,swe_rebench,swe_bench}`; `TRAIN_SOURCE` / `TRAIN_DATASET` / `TRAIN_SEED` select it |

## Scope notes

- Training set `R2E-Gym/R2E-Gym-V1` (Apache-2.0, 8,101 instances), sorted by repository, hence the seeded
  shuffle. SWE-bench Verified is validation only, minus 55 network-dependent instances.
- Environments have no egress. Thinking stays on. Unfinished episodes are masked (`MASK_UNFINISHED_EPISODE`).
- The install step applies guarded fixes to the pinned verl (each refuses to apply if the code changed):
  three for the FP8 rollout of Qwen3.5's HF layout (layer count under `text_config`, module lookup through
  vLLM's name mapper, per-expert split of fused expert tensors), the fully-masked-batch guard, and the
  `hybrid_engine=False` logging call. The FP8 exclusion list is generated from `config.json` and matches
  Qwen's official FP8 checkpoints.
- Only the WorkerPool is configured on the Substrate side. Re-apply after a Substrate reinstall.
- Provider environments are named `ua-*` (`ua-<JOB_SUFFIX>-*` with a suffix); `deploy.sh up` / `down` purge
  this instance's leftovers when `KUBECTL_ATE` is set.
