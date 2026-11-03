"""Every command compiles Warp kernels without NVRTC's precompiled header (a reused header intermittently fails to
map and the compile raises)."""

import sys
from types import ModuleType, SimpleNamespace

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
