"""Guard cached environments against partial or incompatible source updates."""

import hashlib
import importlib.util
from pathlib import Path

import pytest


def patcher():
    path = Path(__file__).resolve().parents[1] / "docker/apply_kamino_patch.py"
    spec = importlib.util.spec_from_file_location("kamino_build_patch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.apply


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "package"
    root.mkdir()
    hashes = {}
    changes = []
    for name in ("first.py", "second.py"):
        (root / name).write_bytes(b"original\n")
        hashes[name] = {
            key: hashlib.sha256(value).hexdigest()
            for key, value in (("before", b"original\n"), ("after", b"optimized\n"))
        }
        changes.append(f"--- a/{name}\n+++ b/{name}\n@@ -1 +1 @@\n-original\n+optimized\n")
    patch = tmp_path / "change.patch"
    patch.write_text("".join(changes))
    return root, patch, hashes


def test_patch_is_idempotent(tree):
    root, patch, hashes = tree
    for _ in range(2):
        patcher()(root, patch, hashes)
        assert all((root / name).read_bytes() == b"optimized\n" for name in hashes)


@pytest.mark.parametrize("second", [b"upstream changed\n", b"optimized\n"])
def test_partial_or_changed_tree_is_rejected_before_any_mutation(tree, second):
    root, patch, hashes = tree
    (root / "second.py").write_bytes(second)
    with pytest.raises(RuntimeError, match="revalidate"):
        patcher()(root, patch, hashes)
    assert (root / "first.py").read_bytes() == b"original\n"
    assert (root / "second.py").read_bytes() == second
