"""Validate container job boundaries without starting Docker or a GPU process."""

import json
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_uv_binaries_match_platform_manifest_and_reject_unknown_architecture(tmp_path):
    spec = importlib.util.spec_from_file_location("install_uv", ROOT / "scripts/install-uv.py")
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    manifest = yaml.safe_load((ROOT / "runtime.lock.yaml").read_text())
    assert installer.VERSION == manifest["uv"]["version"]
    assert set(installer.WHEELS) == {"amd64", "arm64"}
    assert installer.WHEELS["amd64"][1] == manifest["uv"]["linux_x86_64_wheel_sha256"]
    assert installer.WHEELS["arm64"][1] == manifest["uv"]["linux_aarch64_wheel_sha256"]
    with pytest.raises(ValueError, match="unsupported target"):
        installer.install("riscv64", tmp_path)
    assert not list(tmp_path.iterdir())


def test_docker_build_uses_manifest_revision_and_frozen_dependencies():
    manifest = yaml.safe_load((ROOT / "runtime.lock.yaml").read_text())
    dockerfile = (ROOT / "Dockerfile").read_text()
    commits = re.findall(r"^ARG ISAACLAB_COMMIT=([a-f0-9]{40})$", dockerfile, re.MULTILINE)
    assert commits and set(commits) == {manifest["isaaclab"]["revision"]}
    assert "uv sync --frozen" in dockerfile
    assert "runtime.lock.yaml" in dockerfile
    assert "ACCEPT_EULA" not in dockerfile


@pytest.fixture(scope="module")
def compose_config():
    if not shutil.which("docker"):
        pytest.skip("Docker CLI is unavailable; Compose parsing requires it")
    result = subprocess.run(
        ["docker", "compose", "-f", str(ROOT / "compose.yaml"),
         "--profile", "dev", "--profile", "viewer", "config", "--format", "json"],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def test_image_run_and_dev_mount_are_distinct(compose_config):
    services = compose_config["services"]
    assert all("platform" not in config for config in services.values())
    assert {name for name, cfg in services.items() if not cfg.get("profiles")} == {"sim"}
    assert all(v["type"] == "volume" for v in services["sim"]["volumes"])
    dev = services["dev"]
    assert dev["environment"]["HUMANOID_SIM_DEV"] == "1"
    mounts = [v for v in dev["volumes"] if v["type"] == "bind"]
    assert [(v["source"], v["target"]) for v in mounts] == [
        (str(ROOT), "/workspace/humanoid-sim")
    ]


def test_all_compose_jobs_are_finite_and_viewer_is_local(compose_config):
    for cfg in compose_config["services"].values():
        args = cfg["command"]
        assert args[:3] == ["smoke", "--backend", "isaac"]
        assert int(args[args.index("--steps") + 1]) > 0
        if "viewer" not in cfg.get("profiles", []):
            assert "--output-dir" in args
        assert cfg["deploy"]["resources"]["reservations"]["devices"][0]["capabilities"] == ["gpu"]
    viewer = compose_config["services"]["viewer"]
    assert viewer["ports"][0]["host_ip"] == "127.0.0.1"
    assert viewer["command"][viewer["command"].index("--visualizer") + 1] == "viser"
    assert "--realtime" in viewer["command"]
    assert "--output-dir" not in viewer["command"]


def test_sky_uses_required_built_image_and_no_source_sync():
    task = yaml.safe_load((ROOT / "sky/smoke.yaml").read_text())
    assert "workdir" not in task and "file_mounts" not in task
    assert task["envs"]["HUMANOID_SIM_IMAGE"] is None
    assert "${HUMANOID_SIM_IMAGE:?" in task["run"]
    assert "--headless" in task["run"] and "--steps" in task["run"]
    assert "pip install" not in task["setup"] and "git clone" not in task["setup"]
    assert "ports" not in task["resources"]


def test_entrypoint_rejects_invalid_dev_mode_before_python():
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/container-entrypoint.sh"), "--help"],
        env={**os.environ, "HUMANOID_SIM_DEV": "invalid"},
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "HUMANOID_SIM_DEV must be 0 or 1" in result.stderr
