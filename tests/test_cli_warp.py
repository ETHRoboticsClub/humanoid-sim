"""Every command compiles Warp kernels without NVRTC's precompiled header (a reused header intermittently fails to
map and the compile raises)."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from humanoid_sim import cli


def test_commands_run_with_precompiled_headers_off(monkeypatch):
    warp = ModuleType("warp")
    warp.config = SimpleNamespace(use_precompiled_headers=True)
    monkeypatch.setitem(sys.modules, "warp", warp)
    seen = []
    monkeypatch.setattr(cli, "cmd_serve", lambda _args: seen.append(warp.config.use_precompiled_headers) or 0)
    assert cli.main(["serve", "--segment", "unused"]) == 0
    assert seen == [False]


def test_without_warp_installed_commands_still_run(monkeypatch):
    monkeypatch.setitem(sys.modules, "warp", None)  # import warp raises ImportError
    monkeypatch.setattr(cli, "cmd_serve", lambda _args: 0)
    assert cli.main(["serve", "--segment", "unused"]) == 0


def test_retry_passes_after_the_precompiled_header_error():
    calls, sleeps = [], []

    def build():
        calls.append(1)
        if len(calls) < 3:
            raise Exception("catastrophic error: unable to obtain mapped memory")
        return "host"

    assert cli.retry_kernel_build(build, sleep=sleeps.append) == "host"
    assert len(calls) == 3 and len(sleeps) == 2


def test_retry_is_bounded_and_raises_the_last_error():
    calls = []

    def build():
        calls.append(1)
        raise Exception("CUDA kernel build failed with error code 6")

    with pytest.raises(Exception, match="error code 6"):
        cli.retry_kernel_build(build, attempts=3, sleep=lambda _s: None)
    assert len(calls) == 3


def test_other_errors_are_not_retried():
    calls = []

    def build():
        calls.append(1)
        raise ValueError("CUDA kernel build failed with error code 1")

    with pytest.raises(ValueError):
        cli.retry_kernel_build(build, sleep=lambda _s: None)
    assert len(calls) == 1
