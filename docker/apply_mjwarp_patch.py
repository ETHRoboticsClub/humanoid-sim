"""Apply the validated active-row dispatch patch while building the pinned image."""

import subprocess
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

BEFORE = "509b43da297dc6e49efeacdbb59df675213e84afbdb26770471a5ba1855a24b3"
AFTER = "c737461c682ec74fcd9fcd4714f63b063d22bbac1f1c1e71470e1249d11e355f"


def apply(package: Path, patch: Path) -> None:
    source = package / "_src/solver.py"
    if sha256(source.read_bytes()).hexdigest() != BEFORE:
        raise RuntimeError("MJWarp solver differs from the validated 3.12.0 source; revalidate the patch")
    subprocess.run(["git", "apply", "--check", str(patch.resolve())], cwd=package, check=True)
    subprocess.run(["git", "apply", str(patch.resolve())], cwd=package, check=True)
    if sha256(source.read_bytes()).hexdigest() != AFTER:
        raise RuntimeError("MJWarp active-row patch produced an unexpected source hash")


if __name__ == "__main__":
    import mujoco_warp

    if version("mujoco-warp") != "3.12.0":
        raise RuntimeError("The active-row patch requires mujoco-warp 3.12.0")
    apply(
        Path(mujoco_warp.__file__).parent,
        Path(__file__).parent / "patches/mujoco-warp-3.12-active-rows.patch",
    )
