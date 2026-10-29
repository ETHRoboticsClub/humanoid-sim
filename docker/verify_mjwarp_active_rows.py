"""Differential CPU/CUDA checks for the image's active-row MJWarp patch.

Run with the image Python after apply_mjwarp_patch.py. The reference is rebuilt
in a temporary directory by reversing the patch; the installed module is never
modified. Kernel factories bypass name-only memoization to keep sources distinct.
"""

import hashlib
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import warp as wp
from apply_mjwarp_patch import AFTER, BEFORE
from mujoco_warp._src import solver as candidate
from mujoco_warp._src import types


def efc_fixture(worlds, capacity, device):
    rng = np.random.default_rng(218)

    def arr(x, dtype):
        return wp.array(x, dtype=dtype, device=device)

    counts = np.resize(np.array([0, 1, 7, 127, 128, 129, 300, capacity], np.int32), worlds)
    counts = np.minimum(counts, capacity)
    kind = np.full((worlds, capacity), int(types.ConstraintType.LIMIT_JOINT), np.int32)
    if capacity >= 12:
        kind[:, 8:11] = int(types.ConstraintType.CONTACT_ELLIPTIC)
    ids = np.zeros((worlds, capacity), np.int32)
    if capacity >= 12:
        ids[:, 11] = 99
        kind[:, 11] = int(types.ConstraintType.CONTACT_ELLIPTIC)
    done = np.arange(worlds) % 5 == 4
    stale = np.arange(worlds) % 3 == 0
    address = np.array([[8, 9, 10, -1, -1, -1]], np.int32)
    inputs = [
        arr([1.0], wp.float32),
        arr(np.full(worlds, 3, np.int32), wp.int32),
        arr(np.full(worlds, 2, np.int32), wp.int32),
        arr(counts, wp.int32),
        arr([[0.7, 0.6, 0.5, 0.4, 0.3]], types.vec5),
        arr([3], wp.int32),
        arr(address, wp.int32),
        arr(kind, wp.int32),
        arr(ids, wp.int32),
        arr(rng.uniform(0.1, 3, (worlds, capacity)), wp.float32),
        arr(rng.uniform(0, 0.5, (worlds, capacity)), wp.float32),
        arr([1], wp.int32),
        arr(rng.normal(size=(worlds, capacity)), wp.float32),
        arr(stale, wp.bool),
        arr(done, wp.bool),
    ]
    states = rng.integers(0, 4, size=(worlds, capacity), dtype=np.int32)

    def outputs():
        return [
            wp.full((worlds, capacity), -123.0, dtype=wp.float32, device=device),
            arr(states, wp.int32),
            wp.full((worlds, capacity), -1, dtype=wp.int32, device=device),
            wp.zeros(worlds, dtype=wp.int32, device=device),
            wp.zeros(worlds, dtype=wp.int32, device=device),
        ]

    return inputs, outputs


def verify_efc(device):
    # Bypass name-only memoization so the two source modules cannot alias kernels.
    for capacity in (0, 7, 128, 512):
        for track in (False, True):
            inputs, make_outputs = efc_fixture(8, capacity, device)
            old = make_outputs()
            new = make_outputs()
            wp.launch(
                reference._update_constraint_efc.__wrapped__(track),
                dim=(8, capacity),
                inputs=inputs,
                outputs=old,
                device=device,
            )
            wp.launch(
                candidate._update_constraint_efc.__wrapped__(track),
                dim=(8, min(128, capacity)),
                inputs=inputs,
                outputs=new,
                device=device,
            )
            for index in (0, 1, 3, 4):
                np.testing.assert_array_equal(
                    old[index].numpy(), new[index].numpy(), err_msg=f"{device=} {capacity=} {track=} {index=}"
                )
            oldids, newids = old[2].numpy(), new[2].numpy()
            counts = old[3].numpy()
            for world, count in enumerate(counts):
                np.testing.assert_array_equal(np.sort(oldids[world, :count]), np.sort(newids[world, :count]))
    print("REGRESSION_PASS", str(device), flush=True)


def force_fixture(worlds, capacity, device, compact, typical=False):
    rng = np.random.default_rng(918)
    dofs, maxnnz = 33, 8
    counts = np.resize(np.array([0, 1, 7, 127, 128, 129, 300, capacity], np.int32), worlds)
    counts = np.minimum(counts, capacity)
    if typical:
        counts = rng.integers(35, 81, size=worlds, dtype=np.int32)
    nnz = rng.integers(0, maxnnz + 1, size=(worlds, capacity), dtype=np.int32)
    adr = np.broadcast_to(np.arange(capacity, dtype=np.int32) * maxnnz, (worlds, capacity)).copy()
    # Exact binary fractions make accumulation order irrelevant to this comparison.
    values = rng.integers(-8, 9, size=(worlds, 1, capacity * maxnnz)).astype(np.float32) / 8
    colind = rng.integers(0, dofs, size=values.shape, dtype=np.int32)
    force = rng.integers(-4, 5, size=(worlds, capacity)).astype(np.float32) / 4
    mapping = np.broadcast_to(np.arange(dofs, dtype=np.int32), (worlds, dofs)).copy()
    if compact:
        mapping[:, ::3] = -1
        mapping[:, 1::3] = np.arange(dofs // 3, dtype=np.int32)
        mapping[:, 2::3] = np.arange(dofs // 3, dtype=np.int32) + dofs // 3
    changed = (np.arange(worlds) % 4 != 2).astype(np.int32)
    done = np.arange(worlds) % 5 == 4

    def arr(x, dtype):
        return wp.array(x, dtype=dtype, device=device)

    inputs = [
        arr(counts, int),
        arr(nnz, int),
        arr(adr, int),
        arr(colind, int),
        arr(values, float),
        arr(force, float),
        arr(mapping, int),
        arr(changed, int),
        arr(done, bool),
    ]

    def output():
        return wp.full((worlds, dofs), 1.0, dtype=float, device=device)

    return inputs, output


def verify_force(device):
    for capacity in (0, 7, 128, 129, 512):
        for compact in (False, True):
            inputs, make_output = force_fixture(16, capacity, device, compact)
            old, new = make_output(), make_output()
            wp.launch(
                reference._update_constraint_init_qfrc_constraint_sparse.__wrapped__(compact),
                dim=(16, capacity),
                inputs=inputs,
                outputs=[old],
                device=device,
            )
            wp.launch(
                candidate._update_constraint_init_qfrc_constraint_sparse.__wrapped__(compact),
                dim=(16, min(128, capacity)),
                inputs=inputs,
                outputs=[new],
                device=device,
            )
            np.testing.assert_array_equal(
                old.numpy(), new.numpy(), err_msg=f"{device=} {capacity=} {compact=}"
            )
    print("REGRESSION_PASS", str(device), flush=True)


if __name__ == "__main__":
    source = Path(candidate.__file__)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == AFTER
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "_src").mkdir()
        pristine = root / "_src/solver.py"
        shutil.copy2(source, pristine)
        patch = Path(__file__).parent / "patches/mujoco-warp-3.12-active-rows.patch"
        subprocess.run(["git", "apply", "--reverse", str(patch.resolve())], cwd=root, check=True)
        assert hashlib.sha256(pristine.read_bytes()).hexdigest() == BEFORE
        spec = importlib.util.spec_from_file_location("mjwarp_pristine_reference", pristine)
        reference = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = reference
        spec.loader.exec_module(reference)
        for device in ["cpu", *wp.get_cuda_devices()]:
            verify_efc(device)
            verify_force(device)
