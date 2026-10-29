"""Apply the validated Kamino source optimization during environment setup."""

import json
import subprocess
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path


def apply(package: Path, patch: Path, hashes: dict[str, dict[str, str]]) -> None:
    """Check every input before changing any file; accept an already patched tree."""
    actual = {name: sha256((package / name).read_bytes()).hexdigest() for name in hashes}
    if all(actual[name] == expected["after"] for name, expected in hashes.items()):
        return
    if not all(actual[name] == expected["before"] for name, expected in hashes.items()):
        raise RuntimeError("Kamino source differs from the validated Newton 1.6.1 tree; revalidate the patch")
    subprocess.run(["git", "apply", "--check", str(patch.resolve())], cwd=package, check=True)
    subprocess.run(["git", "apply", str(patch.resolve())], cwd=package, check=True)
    for name, expected in hashes.items():
        if sha256((package / name).read_bytes()).hexdigest() != expected["after"]:
            raise RuntimeError(f"Kamino patch produced an unexpected source hash: {name}")


if __name__ == "__main__":
    import newton

    if version("newton") != "1.6.1":
        raise RuntimeError("The Kamino optimization requires Newton 1.6.1")
    root = Path(__file__).parent / "patches"
    for name in ("active-worlds", "active-preconditioner"):
        apply(
            Path(newton.__file__).parent,
            root / f"newton-1.6.1-{name}.patch",
            json.loads((root / f"newton-1.6.1-{name}.json").read_text()),
        )
