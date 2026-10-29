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
"""uni-agent sandbox provider ``ate_env``: one Agent Substrate environment per episode.

The provider implements :class:`uni_agent.sandbox.base.Sandbox` on top of the
``ate_env`` Python client (the Substrate environment API served by ate-env-api).
Each ``start()`` creates an environment from a digest-pinned rootfs image on a base
ActorTemplate; tools run commands with ``Env.start_process`` and read/write files
with the native file RPCs; ``stop()`` deletes the environment.

Long commands (SWE test suites run for many minutes) are collected in bounded
streaming windows and re-attached with byte offsets, so no single gRPC stream
outlives the atenet router's route timeout. ``open_shell`` returns a shell handle
that persists ``cd`` and exported variables across commands without tmux, which
uni-agent would otherwise try to install inside an environment that has no egress.

Config (``sandbox_kwargs`` in the task config, all optional):
  endpoint         ate-env-api address (default ``ate-env-api.ate-env.svc:7777``)
  atespace         atespace of the environments (default ``ate-env``)
  template         base ActorTemplate name (default: the server default template)
  image_digests    JSON file mapping tag refs to digest-pinned refs (default
                   ``$ATE_ENV_IMAGE_DIGESTS``); images already pinned pass through
  actor_prefix     environment id prefix (default ``ua``)
  stream_window_s  max seconds one output stream stays open (default 240)
  auth_token_env   env var holding a bearer token for ate-env-api (default ``SUBSTRATE_ENV_AUTH_TOKEN``, the repo's name)
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import logging
import os
import posixpath
import random
import shlex
import time
import uuid
import weakref
from pathlib import Path
from typing import Any

from ate_env import Client, Env
from ate_env.errors import EnvError, FailedPreconditionError, NotFoundError, RpcError
from ate_env.types import EnvironmentStatus, Signal
from uni_agent.sandbox.base import ExecResult, Sandbox, SandboxConfig
from uni_agent.sandbox.registry import register_sandbox

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "ate-env-api.ate-env.svc:7777"
DEFAULT_ATESPACE = "ate-env"
_ALIVE = {
    EnvironmentStatus.RESUMING,
    EnvironmentStatus.RUNNING,
    EnvironmentStatus.SUSPENDING,
    EnvironmentStatus.SUSPENDED,
    EnvironmentStatus.PAUSING,
    EnvironmentStatus.PAUSED,
}
_STREAM_RETRIES = 3

# One client per (event loop, endpoint, token): uni-agent runs tasks in Ray task processes, each with
# its own loop, and grpc.aio channels are bound to a loop. Keyed on the loop object itself (weakly) so
# a garbage-collected loop takes its clients with it instead of leaving a closed client under a reused id.
_CLIENTS: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[tuple[str, str], Client]] = weakref.WeakKeyDictionary()


def _client_for(endpoint: str, token: str | None) -> Client:
    per_loop = _CLIENTS.setdefault(asyncio.get_running_loop(), {})
    key = (endpoint, token or "")
    client = per_loop.get(key)
    if client is None:
        client = Client(endpoint, auth_token=token) if token else Client(endpoint)
        per_loop[key] = client
    return client


@functools.lru_cache(maxsize=16)
def _load_digest_map(path: str | None) -> dict[str, str]:
    """tag -> digest-pinned reference map; read once per path per process (one sandbox per episode)."""
    if not path:
        return {}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"image digest map {path!r} must be a JSON object")
    return {str(k): str(v) for k, v in data.items()}


def resolve_image(image: str, digests: dict[str, str]) -> str:
    """Return a digest-pinned reference for ``image`` or raise.

    Substrate rejects tag references for ``rootfs_image``; the recipe's
    ``resolve_images.sh`` writes the tag -> digest map this consults.
    """
    if "@sha256:" in image:
        return image
    pinned = digests.get(image)
    if pinned is None and "/" in image and not image.startswith("docker.io/"):
        pinned = digests.get("docker.io/" + image)
    if pinned is None:
        raise RuntimeError(
            f"image {image!r} is not digest-pinned and not in the image digest map; "
            "run integrations/uni-agent/scripts/resolve_images.sh for this dataset slice"
        )
    return pinned


_POOL_FULL_MARKER = "no worker has room"
_SLOW_WAKE_MARKER = "did not answer within the wake budget"
_BUSY_MARKER = "another operation is in progress"


def _is_pool_full(exc: BaseException) -> bool:
    """Substrate could not book the environment on any worker: it is unplaced, so delete it and queue a new one."""
    return _POOL_FULL_MARKER in str(exc)


def _is_slow_wake(exc: BaseException) -> bool:
    """The environment is booked and resuming but missed the server's wake budget (pool under a startup wave).

    Keep waiting on the same environment: deleting it now is rejected while it is mid-resume and would leave
    an orphan holding the booking.
    """
    return _SLOW_WAKE_MARKER in str(exc) and not _is_pool_full(exc)


def _is_capacity_error(exc: BaseException) -> bool:
    """True for either retryable wait condition (pool full or slow wake)."""
    return _is_pool_full(exc) or _is_slow_wake(exc)


_ROUTE_MARKERS = ("did not answer within the running budget", "upstream connect error")
_START_RETRIES = 3


def _is_transient_route_error(exc: BaseException) -> bool:
    """The sandbox is RUNNING but the exec route did not answer (e.g. "upstream connect error" under load):
    the command never started, so starting it again is safe. Wake/capacity errors are handled by start()."""
    text = str(exc)
    return any(m in text for m in _ROUTE_MARKERS) and not _is_capacity_error(exc)


def _is_busy(exc: BaseException) -> bool:
    """gRPC Aborted / "another operation is in progress": the actor is mid-resume or mid-suspend; retry shortly."""
    code = getattr(exc, "code", None)
    return getattr(code, "name", "") == "ABORTED" or _BUSY_MARKER in str(exc)


class _SandboxCoordinator:
    """Per-job coordinator (a detached Ray actor): startup gate plus a registry that reclaims orphans.

    Gate: at most ``limit`` environment start-ups in flight. uni-agent's ``SANDBOX_STARTUP_CONCURRENCY``
    semaphore lives in each runner process (one per session), so it bounds nothing across sessions: a
    1,000-session step issues 1,000 creates and wakes at once. Slots carry a lease so a holder that dies
    without releasing frees its slot.

    Registry: every environment id a provider creates is tracked with a lease that the owning session
    renews while it runs. Ids whose lease expired (owner never got ready, or its process died) or that the
    owner abandoned (delete kept failing) are deleted by a background sweep; under a saturated pool such
    environments otherwise sit RUNNING for the rest of the run, holding bookings no session uses.
    """

    def __init__(self, limit: int, lease_s: float, endpoint: str = "", atespace: str = DEFAULT_ATESPACE,
                 token: str | None = None, recheck_s: float = 5.0, sweep_s: float = 60.0,
                 delete_timeout_s: float = 30.0) -> None:
        self._limit = int(limit)
        self._lease = float(lease_s)
        self._recheck = float(recheck_s)
        self._sweep_s = float(sweep_s)
        self._delete_timeout = float(delete_timeout_s)
        self._endpoint, self._atespace, self._token = endpoint, atespace, token
        self._held: dict[str, float] = {}
        self._cond = asyncio.Condition()
        self._tracked: dict[str, float] = {}   # env id -> lease expiry
        self._due: dict[str, int] = {}         # env id -> delete attempts so far
        self._sweeper: asyncio.Task | None = None
        self._client: Any = None
        self.reclaimed = 0

    # gate
    def _expire(self) -> None:
        now = time.monotonic()
        self._held = {t: e for t, e in self._held.items() if e > now}

    async def acquire(self, token: str) -> None:
        if self._limit <= 0:
            return
        async with self._cond:
            while True:
                self._expire()
                if len(self._held) < self._limit:
                    self._held[token] = time.monotonic() + self._lease
                    return
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._cond.wait(), self._recheck)  # re-check leases periodically

    async def release(self, token: str) -> None:
        async with self._cond:
            if self._held.pop(token, None) is not None:
                self._cond.notify(1)

    def held(self) -> int:
        self._expire()
        return len(self._held)

    # registry
    def track(self, env_id: str, lease_s: float) -> None:
        self._tracked[env_id] = time.monotonic() + float(lease_s)
        self._due.pop(env_id, None)
        self._ensure_sweeper()

    def heartbeat(self, env_id: str, lease_s: float) -> None:
        if env_id in self._tracked:
            self._tracked[env_id] = time.monotonic() + float(lease_s)

    def untrack(self, env_id: str) -> None:
        self._tracked.pop(env_id, None)
        self._due.pop(env_id, None)

    def abandon(self, env_id: str) -> None:
        self._tracked.pop(env_id, None)
        self._due.setdefault(env_id, 0)
        self._ensure_sweeper()

    def pending(self) -> dict[str, int]:
        return dict(self._due)

    def _ensure_sweeper(self) -> None:
        if self._sweeper is None or self._sweeper.done():
            self._sweeper = asyncio.get_running_loop().create_task(self._sweep_loop())

    async def _sweep_loop(self) -> None:
        while True:
            await asyncio.sleep(self._sweep_s)
            with contextlib.suppress(Exception):
                await self.sweep()

    async def sweep(self) -> int:
        """Delete every due environment once; returns how many were reclaimed. Busy ones are retried next round."""
        now = time.monotonic()
        for env_id, expiry in list(self._tracked.items()):
            if expiry <= now:
                self._tracked.pop(env_id, None)
                self._due.setdefault(env_id, 0)
        if not self._due:
            return 0
        if self._client is None:
            self._client = Client(self._endpoint, auth_token=self._token) if self._token else Client(self._endpoint)
        done = 0
        for env_id in list(self._due):
            try:
                await asyncio.wait_for(self._client.delete(env_id, atespace=self._atespace), self._delete_timeout)
            except NotFoundError:
                self._due.pop(env_id, None)  # already gone (owner's delete landed after all, or purged)
                continue
            except FailedPreconditionError as exc:
                # CRASHED actors cannot be suspended, and this API deletes by suspend-then-delete: no retry will
                # help. They hold no booking; deploy.sh purges them with an any-state delete at up/down.
                self._due.pop(env_id, None)
                logger.info("sandbox coordinator: %s cannot be deleted through the API (%s); left for the purge", env_id, exc)
                continue
            except Exception as exc:  # busy (mid-operation) or transient: retry next round, give up after 30 rounds
                self._due[env_id] += 1
                if _is_busy(exc) or self._due[env_id] < 30:
                    continue
                logger.warning("sandbox coordinator: giving up on orphan %s: %s", env_id, exc)
                self._due.pop(env_id, None)
                continue
            self._due.pop(env_id, None)
            done += 1
            self.reclaimed += 1
            logger.warning("sandbox coordinator: reclaimed orphan environment %s", env_id)
        return done


_COORDINATORS: dict[str, Any] = {}


def _coordinator(prefix: str, limit: int, lease_s: float, endpoint: str, atespace: str, token: str | None):
    """Handle to the job-wide coordinator actor, or None outside a Ray worker (then nothing is gated or swept)."""
    try:
        import ray
    except ImportError:
        return None
    if not ray.is_initialized():
        return None
    name = f"{prefix}-sandbox-coordinator"
    handle = _COORDINATORS.get(name)
    if handle is None:
        handle = (
            ray.remote(num_cpus=0, max_concurrency=4096)(_SandboxCoordinator)
            .options(name=name, namespace="uni_agent_substrate", lifetime="detached", get_if_exists=True)
            .remote(limit, lease_s, endpoint, atespace, token)
        )
        _COORDINATORS[name] = handle
    return handle


@register_sandbox("ate_env")
class AteEnvSandbox(Sandbox):
    """Agent Substrate environment as a uni-agent sandbox."""

    supports_shell = True

    def __init__(
        self,
        *,
        image: str,
        runtime_timeout: float = 3600.0,
        endpoint: str = DEFAULT_ENDPOINT,
        atespace: str = DEFAULT_ATESPACE,
        template: str | None = None,
        image_digests: str | None = None,
        actor_prefix: str = "ua",
        stream_window_s: float = 240.0,
        start_wait_s: float = 1800.0,
        startup_concurrency: int | None = None,
        stop_retry_s: float = 120.0,
        auth_token_env: str = "SUBSTRATE_ENV_AUTH_TOKEN",
        **ignored: Any,
    ) -> None:
        if not image:
            raise ValueError("ate_env provider requires a sandbox image")
        if ignored:
            logger.debug("ate_env provider ignoring sandbox kwargs %s", sorted(ignored))
        self.image = image
        self.runtime_timeout = float(runtime_timeout)
        self.endpoint = endpoint
        self.atespace = atespace
        self.template = template or None
        self._digests = _load_digest_map(image_digests or os.environ.get("ATE_ENV_IMAGE_DIGESTS"))
        self.actor_prefix = actor_prefix
        self.stream_window_s = float(stream_window_s)
        self.start_wait_s = float(start_wait_s)
        if startup_concurrency is None:  # default: the same knob uni-agent reads, applied job-wide here
            startup_concurrency = int(os.environ.get("SANDBOX_STARTUP_CONCURRENCY", "0") or 0)
        self.startup_concurrency = int(startup_concurrency)
        self.stop_retry_s = float(stop_retry_s)
        self._token = os.environ.get(auth_token_env) or None
        self._env: Env | None = None
        self.env_id: str | None = None
        self._coord: Any = None
        self._last_heartbeat = 0.0

    @classmethod
    def from_config(cls, config: SandboxConfig) -> AteEnvSandbox:
        if getattr(config, "image_mounts", None):
            raise ValueError("ate_env provider does not support image_mounts; bake tools into the rootfs image")
        return cls(image=config.image or "", runtime_timeout=config.runtime_timeout, **config.sandbox_kwargs)

    # ----- lifecycle -----

    def _coordinator(self):
        if self._coord is None:
            self._coord = _coordinator(self.actor_prefix, self.startup_concurrency, self.start_wait_s + 600.0,
                                       self.endpoint, self.atespace, self._token)
        return self._coord

    async def _registry(self, method: str, *args: Any, wait: bool = True) -> None:
        """Best-effort call on the coordinator's registry (no coordinator outside Ray)."""
        coord = self._coordinator()
        if coord is None:
            return
        ref = getattr(coord, method).remote(*args)
        if wait:
            with contextlib.suppress(Exception):
                await ref

    @contextlib.asynccontextmanager
    async def _startup_slot(self):
        coord = self._coordinator()
        if coord is None or self.startup_concurrency <= 0:
            yield
            return
        token = uuid.uuid4().hex
        await coord.acquire.remote(token)
        try:
            yield
        finally:
            with contextlib.suppress(Exception):
                await coord.release.remote(token)

    async def start(self) -> None:
        """Create an environment and wake it, waiting for the pool instead of failing.

        Two wait conditions, both bounded by ``start_wait_s`` (uni-agent's ``SANDBOX_STARTUP_TIMEOUT`` must
        cover it): the pool is full ("no worker has room": the environment is unplaced, delete it and queue
        a new one with backoff) or the wake missed the server's budget under a startup wave ("did not answer
        within the wake budget" while RESUMING: keep the same environment and probe again; deleting it
        mid-resume is rejected and would leave an orphan holding the booking). Start-ups are also gated
        job-wide by ``startup_concurrency`` (``SANDBOX_STARTUP_CONCURRENCY``), so a step of thousands of
        sessions does not hit the control plane with every create and wake at once. Treating any of this as
        fatal would make ``max_concurrent_sessions`` a hard cap on the sandbox pool.
        """
        client = _client_for(self.endpoint, self._token)
        rootfs = resolve_image(self.image, self._digests)
        deadline = time.monotonic() + self.start_wait_s
        t0 = time.monotonic()
        attempt = 0
        last_log = t0
        async with self._startup_slot():
            while True:
                attempt += 1
                if self._env is None:
                    env_id = f"{self.actor_prefix}-{uuid.uuid4().hex[:12]}"
                    await self._registry("track", env_id, self.start_wait_s + 600.0)  # before create: never untracked
                    self._env = await client.create(
                        env_id, atespace=self.atespace, template_name=self.template, rootfs_image=rootfs
                    )
                    self.env_id = env_id
                try:
                    # First command wakes the environment (cold images pay the template derivation and pull).
                    res = await self._exec(["true"], timeout=min(300.0, self.runtime_timeout))
                    if res.exit_code != 0:
                        raise RuntimeError(
                            f"environment {self.env_id} did not become ready: exit {res.exit_code} {res.stderr[-300:]}"
                        )
                except BaseException as exc:
                    retryable = isinstance(exc, Exception) and _is_capacity_error(exc)
                    if not retryable or time.monotonic() >= deadline:
                        await self.stop()
                        raise
                    if _is_pool_full(exc):
                        await self.stop()  # unplaced: a fresh create queues for the next free booking
                        reason = "pool full"
                    else:
                        reason = f"slow wake, keeping {self.env_id}"
                    if time.monotonic() - last_log >= 60:
                        logger.info("ate_env sandbox: %s, waiting (%.0fs so far, attempt %d)",
                                    reason, time.monotonic() - t0, attempt)
                        last_log = time.monotonic()
                    await asyncio.sleep(min(30.0, 5.0 * attempt) * (0.75 + 0.5 * random.random()))
                    continue
                logger.info("ate_env sandbox %s ready in %.1fs after %d attempt(s) (image %s)",
                            self.env_id, time.monotonic() - t0, attempt, rootfs)
                await self._registry("heartbeat", self.env_id, self._lease_s(), wait=False)
                return

    def _lease_s(self) -> float:
        """Registry lease for a running environment: the whole session budget plus margin, renewed on every
        output window, so only an owner that is really gone lets the coordinator reclaim it."""
        return self.runtime_timeout + 600.0

    async def stop(self) -> None:
        """Delete the environment; retry while Substrate reports it mid-operation (resume/suspend in progress).

        ate-env-api deletes by suspend-then-delete, which is Aborted while the actor is still RESUMING. Giving
        up on the first Aborted leaves the environment RUNNING as an orphan that holds a booking for the rest
        of the run; the actor leaves RESUMING within about a minute, so retry up to ``stop_retry_s``.
        """
        env, self._env = self._env, None
        if env is None:
            return
        deadline = time.monotonic() + self.stop_retry_s
        delay = 2.0
        while True:
            try:
                await env.delete()
            except NotFoundError:
                pass
            except EnvError as exc:
                if _is_busy(exc) and time.monotonic() < deadline:
                    logger.debug("ate_env sandbox %s busy (%s); retrying delete in %.0fs", self.env_id, exc, delay)
                    await asyncio.sleep(delay)
                    delay = min(delay * 2.0, 15.0)
                    continue
                logger.warning("ate_env sandbox %s delete failed: %s (handed to the coordinator sweep)", self.env_id, exc)
                await self._registry("abandon", self.env_id)
                return
            await self._registry("untrack", self.env_id)
            return

    async def is_alive(self) -> bool:
        if self._env is None:
            return False
        try:
            return (await self._env.info()).status in _ALIVE
        except Exception:
            return False

    def _require_env(self) -> Env:
        if self._env is None:
            raise RuntimeError("ate_env sandbox is not started")
        return self._env

    # ----- data plane -----

    async def _exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        limit = float(timeout) if timeout else self.runtime_timeout
        for attempt in range(1, _START_RETRIES + 1):
            try:
                proc = await self._require_env().start_process(list(argv), cwd=workdir or "", env=env, timeout=limit)
                break
            except RpcError as exc:
                if attempt == _START_RETRIES or not _is_transient_route_error(exc):
                    raise
                logger.warning("ate_env start_process for %s did not get through (%s); retrying in %ds",
                               self.env_id, str(exc)[:160], 5 * attempt)
                await asyncio.sleep(5.0 * attempt)
        deadline = time.monotonic() + limit + 5.0
        out, err = bytearray(), bytearray()
        exit_info = None
        retries = 0
        while exit_info is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                await _kill_quietly(proc)
                raise TimeoutError(f"command exceeded {limit:g}s: {argv[:3]}")
            window = min(self.stream_window_s, remaining)
            if time.monotonic() - self._last_heartbeat >= 60.0 and self.env_id:
                self._last_heartbeat = time.monotonic()
                await self._registry("heartbeat", self.env_id, self._lease_s(), wait=False)
            try:
                async with asyncio.timeout(window):
                    async for msg in proc.output(follow=True, stdout_offset=len(out), stderr_offset=len(err)):
                        if msg.stdout:
                            out += msg.stdout
                        if msg.stderr:
                            err += msg.stderr
                        if msg.exit is not None:
                            exit_info = msg.exit
                            break
                retries = 0
            except TimeoutError:
                continue  # window elapsed while the process runs: re-attach with offsets
            except RpcError as exc:
                retries += 1
                if retries > _STREAM_RETRIES:
                    raise
                logger.warning("ate_env output stream for %s interrupted (%s); re-attaching", self.env_id, exc)
                await asyncio.sleep(min(2.0 * retries, 10.0))
        code = exit_info.exit_code
        if code == 137 and time.monotonic() >= deadline - 5.0 - 1.0:
            raise TimeoutError(f"command killed at its {limit:g}s timeout: {argv[:3]}")
        return ExecResult(exit_code=code, stdout=_text(out), stderr=_text(err))

    async def read_file(self, path: str) -> bytes:
        return await self._require_env().read_file_bytes(path)

    async def write_file(self, path: str, content: bytes | str) -> None:
        data = content.encode("utf-8") if isinstance(content, str) else bytes(content)
        parent = posixpath.dirname(path)
        if parent and parent != "/":
            await self._exec(["mkdir", "-p", parent], timeout=30)   # awaited: write_file must not race the mkdir
        await self._require_env().write_file(path, data)

    async def upload_file(self, local_file: Path | str, remote_file: str) -> None:
        await self.write_file(remote_file, Path(local_file).read_bytes())

    async def download_file(self, remote_file: str, local_file: Path | str) -> None:
        data = await self.read_file(remote_file)
        dst = Path(local_file)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)

    async def open_shell(self, *, cwd: str | None = None, env: dict[str, str] | None = None) -> AteEnvShell:
        return AteEnvShell(self, cwd=cwd or "/", env=env or {})


class AteEnvShell:
    """Shell handle for :meth:`AteEnvSandbox.open_shell`.

    Each ``run`` executes one ``bash -c`` command that first restores the working
    directory and exported variables saved by the previous command, then saves them
    again, so ``cd`` and ``export`` persist across tool calls. Background jobs and
    shell functions do not survive between commands.
    """

    def __init__(self, sandbox: AteEnvSandbox, *, cwd: str, env: dict[str, str]) -> None:
        self._sandbox = sandbox
        self._cwd = cwd
        self._env = dict(env)
        self._state = f"/tmp/.ua_shell_{uuid.uuid4().hex[:8]}"

    async def start(self) -> None:
        return None

    async def run(self, command: str, *, timeout: float = 120.0) -> ExecResult:
        state = shlex.quote(self._state)
        script = "\n".join(
            [
                f"__s={state}; mkdir -p \"$__s\"",
                f"cd \"$(cat \"$__s/cwd\" 2>/dev/null || printf %s {shlex.quote(self._cwd)})\" 2>/dev/null || cd /",
                "[ -f \"$__s/env\" ] && . \"$__s/env\" 2>/dev/null",
                "unset __s",
                command,
                "__rc=$?",
                f"pwd > {state}/cwd",
                f"declare -px 2>/dev/null | grep -vE '^declare -x (PWD|OLDPWD|SHLVL|_|__rc|__s)=' > {state}/env",
                "exit $__rc",
            ]
        )
        return await self._sandbox.exec(["bash", "-c", script], timeout=timeout, env=self._env or None)

    async def close(self) -> None:
        try:
            await self._sandbox.exec(["rm", "-rf", self._state], timeout=30)
        except Exception:  # best effort; the environment is deleted at stop()
            pass


def _text(data: bytearray) -> str:
    return bytes(data).decode("utf-8", errors="replace")


async def _kill_quietly(proc: Any) -> None:
    """SIGKILL without waiting: the guest reaps the process, and waiting would reopen a stream."""
    try:
        await proc.signal(Signal.KILL)
    except Exception:
        pass
