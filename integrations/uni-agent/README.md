# uni-agent on Agent Substrate

A [uni-agent](https://github.com/verl-project/uni-agent) sandbox provider that runs
each episode in an Agent Substrate environment through `ate-env-api`, an `r2e_gym`
task that grades the episode with hidden tests in the same environment, and the
tooling that turns SWE datasets into uni-agent task rows with digest-pinned images.
Used by `examples/Qwen3.5-35B-A3B-verl`.

## Install

```bash
pip install ./clients/python             # ate_env client
# uni-agent is not on PyPI: install the pinned commit without its dependency list (it declares none that
# matter here beyond the five below), then this package
git clone --filter=blob:none https://github.com/verl-project/uni-agent.git && git -C uni-agent checkout 174b040
pip install --no-deps ./uni-agent && pip install pydantic pyyaml httpx jsonschema openai
pip install ./integrations/uni-agent     # this package
# make every Python process register the provider and task with uni-agent:
echo "import uni_agent_substrate._register" > "$(python3 -c 'import site;print(site.getsitepackages()[0])')/uni_agent_substrate.pth"
```

Check: `python3 -c "from uni_agent.sandbox import get_sandbox_cls; print(get_sandbox_cls('ate_env'))"`.

## Provider `ate_env`

Task config entry:

```yaml
- name: r2e_gym
  sandbox:
    provider: ate_env
    runtime_timeout: 3600
    sandbox_kwargs:
      endpoint: ate-env-api.ate-env.svc:7777
      atespace: ate-env
      template: swe-2c6g              # base ActorTemplate; derived per rootfs image
      image_digests: /workspace/agent/images.json   # tag -> digest-pinned ref
```

The row's `sandbox.image` must be digest-pinned or present in `image_digests`
(Substrate rejects tags). Commands run through `Env.start_process` with output
collected in `stream_window_s` windows (default 240 s) so long test runs survive the
router's stream timeout; `open_shell` keeps `cd` and exports across commands without
tmux. Environments have no egress. Inside Ray, one detached actor per job,
`<actor_prefix>-sandbox-coordinator`, gates environment start-ups job-wide at
`SANDBOX_STARTUP_CONCURRENCY` (also `startup_concurrency` in `sandbox_kwargs`; uni-agent's own
semaphore is per process) and tracks every environment id with a lease the owner renews: ids whose
owner never got ready, died, or could not delete them are deleted by its sweep, so no orphan keeps a
booking. `start_wait_s` bounds how long a session waits for the pool (full pool: recreate; slow wake:
same environment); `stop_retry_s` (120) retries deletes while the actor is mid-operation.

## Task `r2e_gym`

Metadata needs `expected_output_json`, `parsed_commit_content` (oracle only) and the
image on `sandbox.image`. Config: `eval_timeout` (900), `setup_timeout` (300),
`run_oracle_solution` (apply the gold patch instead of running the agent; must score 1.0).

## Datasets and images

```bash
pip install -e './integrations/uni-agent[datasets]'
python3 -m uni_agent_substrate.datasets --source r2e_gym_easy --max-instances 64 --images-out images.txt --out train.parquet
integrations/uni-agent/scripts/resolve_images.sh images.txt <region>-docker.pkg.dev/<project>/<dockerhub-remote-repo> images.json
python3 -m uni_agent_substrate.datasets --source r2e_gym_easy --max-instances 64 --image-digests images.json --out train.parquet
```

`datasets` (`--source r2e_gym_easy | r2e_gym | swe_rebench | swe_bench`, `--dataset`
to override the Hugging Face id, e.g. `R2E-Gym/R2E-Gym-V1`, `--split`, `--max-instances`,
`--shuffle-seed` for datasets sorted by repository, `--exclude-file`) emits
one row shape for every source: `prompt`, `task_name`, `sandbox.image` and the
instance metadata the task needs. Substrate only accepts digest-pinned images, so
`resolve_images.sh` resolves the tag references (`--images-out`) through an Artifact
Registry remote repository proxying Docker Hub (`crane digest`, nothing copied) or a
mirror (`MODE=mirror`), and `--image-digests` rewrites the rows to those references.

## Tests

```bash
pip install -e './integrations/uni-agent[test]' && pytest integrations/uni-agent/tests
```

Tests use the fake environment/guest services from `clients/python/tests/fakes.py`.
