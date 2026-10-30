"""Apply the validated Kamino source optimization during environment setup."""

import json
import subprocess
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path


def _digest(path: Path) -> str | None:
    """An absent source is represented explicitly for patches that add files."""
    return sha256(path.read_bytes()).hexdigest() if path.exists() else None


def apply(package: Path, patch: Path, hashes: dict[str, dict[str, str | None]]) -> None:
    """Check every input before changing any file; accept an already patched tree."""
    actual = {name: _digest(package / name) for name in hashes}
    if all(actual[name] == expected["after"] for name, expected in hashes.items()):
        return
    if not all(actual[name] == expected["before"] for name, expected in hashes.items()):
        raise RuntimeError("Kamino source differs from the validated Newton 1.6.1 tree; revalidate the patch")
    subprocess.run(["git", "apply", "--check", str(patch.resolve())], cwd=package, check=True)
    subprocess.run(["git", "apply", str(patch.resolve())], cwd=package, check=True)
    for name, expected in hashes.items():
        if _digest(package / name) != expected["after"]:
            raise RuntimeError(f"Kamino patch produced an unexpected source hash: {name}")


def apply_sequence(package: Path, patches: list[tuple[Path, dict]]) -> None:
    """Match a complete validated prefix, including files touched by later stages.

    Per-file membership in a list of known hashes is insufficient: a mixed tree
    can contain valid individual hashes without representing any valid prefix.
    Validate the whole chain and current prefix before the first write.
    """
    initial = {}
    for _, hashes in patches:
        for name, values in hashes.items():
            initial.setdefault(name, values["before"])
    states = [initial]
    for _, hashes in patches:
        previous = states[-1]
        if any(previous[name] != values["before"] for name, values in hashes.items()):
            raise RuntimeError("Kamino patch manifests do not form a validated chain; revalidate the patch")
        states.append(previous | {name: values["after"] for name, values in hashes.items()})
    actual = {name: _digest(package / name) for name in initial}
    prefixes = [index for index, state in enumerate(states) if actual == state]
    if not prefixes:
        raise RuntimeError("Kamino source is not a validated patch prefix; revalidate the patch")
    for patch, hashes in patches[max(prefixes):]:
        apply(package, patch, hashes)


if __name__ == "__main__":
    import newton

    if version("newton") != "1.6.1":
        raise RuntimeError("The Kamino optimization requires Newton 1.6.1")
    root = Path(__file__).parent / "patches"
    apply_sequence(Path(newton.__file__).parent, [
        (root / f"newton-1.6.1-{name}.patch", json.loads((root / f"newton-1.6.1-{name}.json").read_text()))
        for name in ("active-worlds", "active-preconditioner", "register-budget", "body-assembly", "tracked-active-clear", "bilateral-reduced", "reduced-runtime", "runtime-throughput", "lazy-preconditioning", "factor-pattern-guard")
    ])
