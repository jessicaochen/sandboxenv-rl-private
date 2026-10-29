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
"""Provider tests against the in-process fake Substrate environment API (clients/python/tests/fakes.py)."""

from __future__ import annotations

import json

import pytest

from uni_agent_substrate import provider as prov
from uni_agent_substrate.provider import AteEnvSandbox, resolve_image

from .fakes import FakeProc

PINNED = "registry.example/repo/numpy_final@sha256:" + "a" * 64


def _sandbox(fake_client, **kw) -> AteEnvSandbox:
    sb = AteEnvSandbox(image=PINNED, endpoint="fake", **kw)
    import asyncio

    prov._CLIENTS.clear()
    prov._CLIENTS[asyncio.get_running_loop()] = {("fake", ""): fake_client}
    return sb


def test_resolve_image_requires_digest(tmp_path):
    with pytest.raises(RuntimeError):
        resolve_image("namanjain12/numpy_final:abc", {})
    assert resolve_image(PINNED, {}) == PINNED
    assert resolve_image("namanjain12/numpy_final:abc", {"namanjain12/numpy_final:abc": PINNED}) == PINNED
    assert resolve_image("namanjain12/numpy_final:abc", {"docker.io/namanjain12/numpy_final:abc": PINNED}) == PINNED


async def test_start_creates_pinned_env_and_probes(fake_stack):
    client, fakes = fake_stack
    sb = _sandbox(client, template="swe-2c6g", atespace="ate-env")
    fakes.processes.next_procs.append(FakeProc(exit_code=0))
    await sb.start()
    assert sb.env_id and sb.env_id.startswith("ua-")
    env = fakes.environments.environments[("ate-env", sb.env_id)]
    assert env.template.name.startswith("swe-2c6g-")  # derived template from the digest
    probe = next(iter(fakes.processes.procs.values()))
    assert probe.command == ["true"]
    assert await sb.is_alive()
    await sb.stop()
    assert ("ate-env", sb.env_id) not in fakes.environments.environments
    await sb.stop()  # idempotent


async def test_failed_start_deletes_the_environment(fake_stack):
    client, fakes = fake_stack
    sb = _sandbox(client, template="swe-2c6g", atespace="ate-env")
    fakes.processes.next_procs.append(FakeProc(exit_code=7))  # the wake-up probe fails
    with pytest.raises(RuntimeError):
        await sb.start()
    assert not any(k[1].startswith("ua-") for k in fakes.environments.environments)  # nothing left booked
    assert sb._env is None


def _capacity_error():
    from ate_env.errors import RpcError

    return RpcError("environment ate-env/ua-x did not answer within the wake budget of 2m0s while SUSPENDED (19 attempts): "
                    "rpc error: code = Unavailable desc = actor ate-env/ua-x unavailable: no worker has room for the actor")


async def test_start_waits_for_capacity_then_succeeds(fake_stack, monkeypatch):
    client, fakes = fake_stack
    sb = _sandbox(client, template="swe-2c6g", atespace="ate-env", start_wait_s=60)
    monkeypatch.setattr(prov.asyncio, "sleep", _no_sleep(monkeypatch))
    calls = {"n": 0}
    real_exec = sb._exec

    async def flaky_exec(argv, **kw):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _capacity_error()
        return await real_exec(argv, **kw)

    monkeypatch.setattr(sb, "_exec", flaky_exec)
    fakes.processes.next_procs.append(FakeProc(exit_code=0))
    await sb.start()
    assert calls["n"] == 3 and sb.env_id is not None
    live = [k for k in fakes.environments.environments if k[1].startswith("ua-")]
    assert live == [("ate-env", sb.env_id)]  # the two environments that never woke were deleted
    await sb.stop()


async def test_start_gives_up_on_capacity_after_start_wait(fake_stack, monkeypatch):
    client, fakes = fake_stack
    sb = _sandbox(client, template="swe-2c6g", atespace="ate-env", start_wait_s=0)

    async def always_full(argv, **kw):
        raise _capacity_error()

    monkeypatch.setattr(sb, "_exec", always_full)
    from ate_env.errors import RpcError

    with pytest.raises(RpcError):
        await sb.start()
    assert not any(k[1].startswith("ua-") for k in fakes.environments.environments)


async def test_start_does_not_retry_non_capacity_errors(fake_stack, monkeypatch):
    client, fakes = fake_stack
    sb = _sandbox(client, template="swe-2c6g", atespace="ate-env", start_wait_s=60)
    calls = {"n": 0}

    async def broken(argv, **kw):
        calls["n"] += 1
        raise RuntimeError("guest crashed")

    monkeypatch.setattr(sb, "_exec", broken)
    with pytest.raises(RuntimeError):
        await sb.start()
    assert calls["n"] == 1
    assert not any(k[1].startswith("ua-") for k in fakes.environments.environments)


def _slow_wake_error():
    from ate_env.errors import RpcError

    return RpcError("environment ate-env/ua-x did not answer within the wake budget of 2m0s while RESUMING (17 attempts): "
                    "rpc error: code = Unavailable desc = ")


async def test_slow_wake_keeps_the_same_environment(fake_stack, monkeypatch):
    client, fakes = fake_stack
    sb = _sandbox(client, template="swe-2c6g", atespace="ate-env", start_wait_s=60)
    monkeypatch.setattr(prov.asyncio, "sleep", _no_sleep(monkeypatch))
    calls = {"n": 0}
    real_exec = sb._exec

    async def slow_exec(argv, **kw):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _slow_wake_error()
        return await real_exec(argv, **kw)

    monkeypatch.setattr(sb, "_exec", slow_exec)
    fakes.processes.next_procs.append(FakeProc(exit_code=0))
    created = len(fakes.environments.environments)
    await sb.start()
    assert calls["n"] == 3
    assert len(fakes.environments.environments) == created + 1  # one environment, probed three times, never deleted
    assert ("ate-env", sb.env_id) in fakes.environments.environments
    await sb.stop()


async def test_slow_wake_gives_up_at_start_wait(fake_stack, monkeypatch):
    client, fakes = fake_stack
    sb = _sandbox(client, template="swe-2c6g", atespace="ate-env", start_wait_s=0)

    async def never_wakes(argv, **kw):
        raise _slow_wake_error()

    monkeypatch.setattr(sb, "_exec", never_wakes)
    from ate_env.errors import RpcError

    with pytest.raises(RpcError):
        await sb.start()
    assert not any(k[1].startswith("ua-") for k in fakes.environments.environments)  # deleted on giving up


def _busy_error():
    import grpc
    from ate_env.errors import RpcError

    return RpcError('ate: suspending "ua-x": rpc error: code = Aborted desc = another operation is in progress for this actor',
                    grpc.StatusCode.ABORTED)


async def test_stop_retries_while_the_actor_is_busy(fake_stack, monkeypatch):
    client, fakes = fake_stack
    sb = _sandbox(client, template="swe-2c6g", atespace="ate-env")
    monkeypatch.setattr(prov.asyncio, "sleep", _no_sleep(monkeypatch))
    fakes.processes.next_procs.append(FakeProc(exit_code=0))
    await sb.start()
    env, calls = sb._env, {"n": 0}
    real_delete = env.delete

    async def busy_then_ok():
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _busy_error()
        await real_delete()

    monkeypatch.setattr(env, "delete", busy_then_ok)
    await sb.stop()
    assert calls["n"] == 3
    assert ("ate-env", sb.env_id) not in fakes.environments.environments


async def test_stop_gives_up_after_stop_retry_budget(fake_stack, monkeypatch, caplog):
    client, fakes = fake_stack
    sb = _sandbox(client, template="swe-2c6g", atespace="ate-env", stop_retry_s=0)
    fakes.processes.next_procs.append(FakeProc(exit_code=0))
    await sb.start()
    env, calls = sb._env, {"n": 0}

    async def always_busy():
        calls["n"] += 1
        raise _busy_error()

    monkeypatch.setattr(env, "delete", always_busy)
    with caplog.at_level("WARNING"):
        await sb.stop()
    assert calls["n"] == 1 and sb._env is None
    assert "delete failed" in caplog.text


async def test_start_gate_bounds_concurrency_and_expires_leases():
    import asyncio

    gate = prov._SandboxCoordinator(limit=2, lease_s=0.05, recheck_s=0.01)
    await gate.acquire("a")
    await gate.acquire("b")
    third = asyncio.create_task(gate.acquire("c"))
    await asyncio.sleep(0)
    assert not third.done() and gate.held() == 2
    await gate.release("a")
    await asyncio.wait_for(third, 1.0)  # released slot handed over
    assert gate.held() == 2
    await asyncio.sleep(0.08)  # leases expire: a crashed holder frees its slot
    assert gate.held() == 0
    await gate.acquire("d")
    assert gate.held() == 1


class _FakeDeleteClient:
    def __init__(self, fail=None):
        self.deleted, self.fail = [], dict(fail or {})

    async def delete(self, env_id, *, atespace):
        exc = self.fail.get(env_id)
        if exc:
            if isinstance(exc, list):
                if exc:
                    raise exc.pop(0)
            else:
                raise exc
        self.deleted.append((atespace, env_id))


async def test_coordinator_reclaims_expired_and_abandoned_environments():
    import asyncio

    from ate_env.errors import FailedPreconditionError, NotFoundError

    coord = prov._SandboxCoordinator(limit=0, lease_s=1, atespace="ate-env", sweep_s=3600)
    client = coord._client = _FakeDeleteClient(fail={"ua-busy": [_busy_error()], "ua-gone": NotFoundError("gone")})
    coord.track("ua-live", 60)            # owner alive: never touched
    coord.track("ua-stale", 0)            # lease expired: owner never got ready or died
    coord.track("ua-busy", 0)
    coord.track("ua-gone", 0)
    coord.track("ua-kept", 0)
    coord.heartbeat("ua-kept", 60)        # renewed in time
    coord.abandon("ua-failed")            # delete failed in stop(): due immediately
    coord.abandon("ua-crashed")           # CRASHED actor: suspend prerequisite fails for good
    client.fail["ua-crashed"] = FailedPreconditionError("MarkSuspending prerequisite not met")
    coord.untrack("ua-done")              # cleanly deleted by its owner
    await asyncio.sleep(0)
    coord._sweeper.cancel()
    assert await coord.sweep() == 2
    assert sorted(e for _, e in client.deleted) == ["ua-failed", "ua-stale"]
    assert coord.pending() == {"ua-busy": 1}   # mid-operation: retried next round; the CRASHED one was dropped at once
    assert await coord.sweep() == 1 and coord.pending() == {}
    assert "ua-live" in coord._tracked and "ua-kept" in coord._tracked


async def test_coordinator_sweep_bounds_each_delete():
    import asyncio

    class Hanging:
        async def delete(self, env_id, *, atespace):
            await asyncio.sleep(3600)

    coord = prov._SandboxCoordinator(limit=0, lease_s=1, sweep_s=3600, delete_timeout_s=0.01)
    coord._client = Hanging()
    coord.abandon("ua-hang")
    coord._sweeper.cancel()
    assert await asyncio.wait_for(coord.sweep(), 2.0) == 0   # a hung control plane cannot wedge the sweeper
    assert coord.pending() == {"ua-hang": 1}


async def test_coordinator_gate_disabled_when_limit_is_zero():
    coord = prov._SandboxCoordinator(limit=0, lease_s=1)
    await coord.acquire("a")
    await coord.acquire("b")
    assert coord.held() == 0


def _route_error():
    import grpc
    from ate_env.errors import RpcError

    return RpcError("environment ate-env/ua-x did not answer within the running budget of 20s while RUNNING (7 attempts): "
                    "rpc error: code = Unavailable desc = upstream connect error or disconnect/reset before headers",
                    grpc.StatusCode.UNAVAILABLE)


async def test_exec_retries_a_transient_route_error_at_start(fake_stack, monkeypatch):
    client, fakes = fake_stack
    sb = _sandbox(client, template="swe-2c6g", atespace="ate-env")
    monkeypatch.setattr(prov.asyncio, "sleep", _no_sleep(monkeypatch))
    fakes.processes.next_procs.append(FakeProc(exit_code=0))
    await sb.start()
    env, calls = sb._env, {"n": 0}
    real_start = env.start_process

    async def flaky_start(*a, **kw):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _route_error()
        return await real_start(*a, **kw)

    monkeypatch.setattr(env, "start_process", flaky_start)
    fakes.processes.next_procs.append(FakeProc(chunks=[("stdout", b"ok\n")], exit_code=0))
    res = await sb._exec(["echo", "ok"])
    assert calls["n"] == 3 and res.exit_code == 0 and res.stdout == "ok\n"

    async def always_down(*a, **kw):
        raise _route_error()

    monkeypatch.setattr(env, "start_process", always_down)
    from ate_env.errors import RpcError

    with pytest.raises(RpcError):
        await sb._exec(["true"])
    await sb.stop()


def test_error_classification():
    from ate_env.errors import RpcError

    assert prov._is_pool_full(_capacity_error()) and not prov._is_slow_wake(_capacity_error())
    assert prov._is_slow_wake(_slow_wake_error()) and not prov._is_pool_full(_slow_wake_error())
    assert prov._is_busy(_busy_error()) and prov._is_busy(RpcError("another operation is in progress for this actor"))
    assert not prov._is_busy(RuntimeError("guest crashed"))
    assert prov._is_transient_route_error(_route_error()) and not prov._is_transient_route_error(_slow_wake_error())


def _no_sleep(monkeypatch):
    async def fake_sleep(_seconds):
        return None

    return fake_sleep


async def test_exec_collects_output_and_exit(fake_stack):
    client, fakes = fake_stack
    sb = _sandbox(client)
    fakes.processes.next_procs.append(FakeProc(exit_code=0))
    await sb.start()
    fakes.processes.next_procs.append(FakeProc(chunks=[("stdout", b"hello "), ("stderr", b"warn"), ("stdout", b"world")], exit_code=3))
    res = await sb.exec(["sh", "-c", "x"], workdir="/testbed", env={"A": "1"}, timeout=10)
    assert (res.exit_code, res.stdout, res.stderr) == (3, "hello world", "warn")
    started = list(fakes.processes.procs.values())[-1]
    assert started.cwd == "/testbed" and started.env == {"A": "1"} and started.timeout_seconds == 10


async def test_exec_reattaches_after_stream_window(fake_stack):
    client, fakes = fake_stack
    sb = _sandbox(client, stream_window_s=0.2)
    fakes.processes.next_procs.append(FakeProc(exit_code=0))
    await sb.start()
    # running_polls keeps the process alive across follow streams; the fake ends a follow
    # stream only after replaying chunks, so a window that expires first forces a re-attach.
    fakes.processes.next_procs.append(FakeProc(chunks=[("stdout", b"part1-"), ("stdout", b"part2")], exit_code=0, hang_follow=True))
    proc_script = fakes.processes.next_procs[-1]

    async def unhang():
        import asyncio

        await asyncio.sleep(0.5)
        proc_script.hang_follow = False

    import asyncio

    asyncio.get_running_loop().create_task(unhang())
    res = await sb.exec(["long"], timeout=10)
    assert res.exit_code == 0
    assert res.stdout.count("part1-") == 1 and res.stdout.endswith("part2")


async def test_exec_timeout_returns_minus_one(fake_stack):
    client, fakes = fake_stack
    sb = _sandbox(client, stream_window_s=0.2)
    fakes.processes.next_procs.append(FakeProc(exit_code=0))
    await sb.start()
    fakes.processes.next_procs.append(FakeProc(hang_follow=True))
    res = await sb.exec(["sleep", "999"], timeout=0.3)
    assert res.exit_code == -1
    proc = list(fakes.processes.procs.values())[-1]
    assert 9 in proc.signals  # killed


async def test_files_roundtrip(fake_stack):
    client, fakes = fake_stack
    sb = _sandbox(client)
    fakes.processes.next_procs.append(FakeProc(exit_code=0))
    await sb.start()
    await sb.write_file("/tmp/dir/a.txt", "héllo")
    assert await sb.read_file("/tmp/dir/a.txt") == "héllo".encode()
    mk = [p for p in fakes.processes.procs.values() if p.command[:2] == ["mkdir", "-p"]]
    assert mk and mk[-1].command[2] == "/tmp/dir"


async def test_shell_persists_state_script(fake_stack):
    client, fakes = fake_stack
    sb = _sandbox(client)
    fakes.processes.next_procs.append(FakeProc(exit_code=0))
    await sb.start()
    shell = await sb.open_shell(cwd="/testbed", env={"FOO": "bar"})
    fakes.processes.next_procs.append(FakeProc(chunks=[("stdout", b"ok")], exit_code=0))
    res = await shell.run("cd src && export X=1", timeout=5)
    assert res.exit_code == 0 and res.stdout == "ok"
    proc = list(fakes.processes.procs.values())[-1]
    assert proc.command[:2] == ["bash", "-c"] and proc.env == {"FOO": "bar"}
    script = proc.command[2]
    assert "cd src && export X=1" in script and "declare -px" in script and "/testbed" in script
    await shell.close()


def test_from_config_rejects_image_mounts():
    from uni_agent.sandbox.base import SandboxConfig

    cfg = SandboxConfig(provider="ate_env", image=PINNED, sandbox_kwargs={"template": "swe-2c6g"})
    sb = AteEnvSandbox.from_config(cfg)
    assert sb.template == "swe-2c6g" and sb.image == PINNED
    assert json.dumps(sb._digests) == "{}"
