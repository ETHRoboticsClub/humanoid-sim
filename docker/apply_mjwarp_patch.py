"""Apply source-guarded MJWarp optimizations while building the pinned image."""

import subprocess
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

BEFORE = "509b43da297dc6e49efeacdbb59df675213e84afbdb26770471a5ba1855a24b3"
AFTER = "c737461c682ec74fcd9fcd4714f63b063d22bbac1f1c1e71470e1249d11e355f"
SENSOR_BEFORE = "f5c0793d97374c7c58e1b8f065ab1386069875a8fd39c1aebdd4ce861b3526d3"
SENSOR_AFTER = "1b64d41dac172b7ba6488bfe69a22dfc90363907d7fd9c971a1123f5a02d5eb6"
FORWARD_BEFORE = "7b7ffa3d914b6adf65f228261e207998d5bed1e59d4fb17f5bc6ac796aacf490"
FORWARD_AFTER = "6f4220103cbb0605bb9f1201d22da9e0c2f3fb3ac985fa5c4666e77f24acdc14"


def apply(package: Path, patch: Path, *, source_name="solver.py", before=BEFORE, after=AFTER) -> None:
    source = package / "_src" / source_name
    if sha256(source.read_bytes()).hexdigest() != before:
        raise RuntimeError(f"MJWarp {source_name} differs from the validated 3.12.0 source; revalidate the patch")
    subprocess.run(["git", "apply", "--check", str(patch.resolve())], cwd=package, check=True)
    subprocess.run(["git", "apply", str(patch.resolve())], cwd=package, check=True)
    if sha256(source.read_bytes()).hexdigest() != after:
        raise RuntimeError(f"MJWarp {source_name} patch produced an unexpected source hash")


if __name__ == "__main__":
    import mujoco_warp

    if version("mujoco-warp") != "3.12.0":
        raise RuntimeError("The patches require mujoco-warp 3.12.0")
    apply(
        Path(mujoco_warp.__file__).parent,
        Path(__file__).parent / "patches/mujoco-warp-3.12-active-rows.patch",
    )
    apply(
        Path(mujoco_warp.__file__).parent,
        Path(__file__).parent / "patches/mujoco-warp-3.12-no-unused-tactile.patch",
        source_name="sensor.py", before=SENSOR_BEFORE, after=SENSOR_AFTER,
    )
    apply(
        Path(mujoco_warp.__file__).parent,
        Path(__file__).parent / "patches/mujoco-warp-3.12-rk4-sensor-stages.patch",
        source_name="forward.py", before=FORWARD_BEFORE, after=FORWARD_AFTER,
    )
