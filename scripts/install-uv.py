"""Install a checksum-pinned uv binary for the Docker target architecture."""
import hashlib
import io
from pathlib import Path
import sys
import urllib.request
import zipfile

VERSION = "0.12.9"
WHEELS = {
    "amd64": (
        "https://files.pythonhosted.org/packages/37/4b/cd04809c7ad5149faac55160925bd67aab43f803f6c01be1c32aec7d24d9/uv-0.12.9-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "5badfd805fd88bf99b4b4f044f6e8f762f1892cab27477f4427bb473e93dd049",
    ),
    "arm64": (
        "https://files.pythonhosted.org/packages/cc/29/649ff920dcd2373a819a4e0d9c1989ae770698e634c277e5a6fbbe19e9b4/uv-0.12.9-py3-none-manylinux_2_17_aarch64.manylinux2014_aarch64.musllinux_1_1_aarch64.whl",
        "cce6e716df290b7f4a31f76ef39866e1dc006e947520a660459dea00b5d55234",
    ),
}


def install(architecture: str, destination: Path = Path("/usr/local/bin")) -> None:
    if architecture not in WHEELS:
        raise ValueError(f"unsupported target architecture: {architecture!r}; expected amd64 or arm64")
    url, expected = WHEELS[architecture]
    wheel = urllib.request.urlopen(url, timeout=120).read()
    if hashlib.sha256(wheel).hexdigest() != expected:
        raise RuntimeError("uv wheel checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
        for name in ("uv", "uvx"):
            path = destination / name
            path.write_bytes(archive.read(f"uv-{VERSION}.data/scripts/{name}"))
            path.chmod(0o755)


if __name__ == "__main__":
    install(sys.argv[1])
