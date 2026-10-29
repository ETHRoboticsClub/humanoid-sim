"""A cached SkyPilot environment must retain verified patches across setup runs."""

import hashlib
import importlib.util
from pathlib import Path

import pytest


def patcher():
    path = Path(__file__).resolve().parents[1] / "docker/apply_mjwarp_patch.py"
    spec = importlib.util.spec_from_file_location("mjwarp_build_patch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(tmp_path):
    root = tmp_path / "package"
    (root / "_src").mkdir(parents=True)
    source = root / "_src/solver.py"
    original, patched = b"original\n", b"optimized\n"
    source.write_bytes(original)
    patch = tmp_path / "change.patch"
    patch.write_text("--- a/_src/solver.py\n+++ b/_src/solver.py\n@@ -1 +1 @@\n-original\n+optimized\n")
    hashes = {"before": hashlib.sha256(original).hexdigest(), "after": hashlib.sha256(patched).hexdigest()}
    return root, source, patch, hashes


def test_reapplying_verified_patch_preserves_source(tmp_path):
    root, source, patch, hashes = fixture(tmp_path)
    apply = patcher().apply
    apply(root, patch, **hashes)
    first = source.read_bytes()
    apply(root, patch, **hashes)
    assert source.read_bytes() == first == b"optimized\n"


def test_patch_rejects_unrecognized_source_without_modifying_it(tmp_path):
    root, source, patch, hashes = fixture(tmp_path)
    source.write_bytes(b"changed upstream\n")
    with pytest.raises(RuntimeError, match="revalidate"):
        patcher().apply(root, patch, **hashes)
    assert source.read_bytes() == b"changed upstream\n"
