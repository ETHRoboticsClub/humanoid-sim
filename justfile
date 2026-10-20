set shell := ["bash", "-euo", "pipefail", "-c"]

image := env("HSIM_IMAGE", "humanoid-sim:dev")
# The runtime segment lives on this Docker volume; Controls' sim stack mounts the same one.
volume := env("HSIM_VOLUME", "hsim-run")
# A GPU when the host has one (the sim keeps up with real time only there), else Warp's CPU device.
detected := `command -v nvidia-smi >/dev/null 2>&1 && echo cuda:0 || echo cpu`
device := env("HSIM_DEVICE", detected)
gpus := if device =~ "^cuda" { "--gpus all" } else { "" }
run := "docker run --rm " + gpus + " -v \"$PWD\":/workspace/humanoid-sim -v " + volume + ":/run/hsim -v hsim-cache:/root/.cache -w /workspace/humanoid-sim " + image

default:
    @just --list

# Contract tests, natively.
test *args:
    uv run pytest -m "not physics" {{args}}

# The Isaac Lab image, for this machine's architecture.
image:
    docker buildx build --load -f docker/Dockerfile -t {{image}} .

# The robot's USD asset (once per description or profile change).
asset robot="ethrc_v1":
    {{run}} --robot {{robot}} build-asset

# The robot behind the runtime segment `<volume>:/plant`, serving client after client (Controls' plant host).
serve robot="ethrc_v1" support="suspended" *args:
    docker volume create {{volume}} >/dev/null
    {{run}} --robot {{robot}} serve --segment /run/hsim/plant --device {{device}} --support {{support}} --persist --truth-every 1 --client-timeout-ms 10000 {{args}}

# A Rerun recording of the running host, written to outputs/runtime.rrd.
view robot="ethrc_v1":
    {{run}} --robot {{robot}} view-runtime /run/hsim/plant --save outputs/runtime.rrd

# Physics tests in the image.
test-physics *args:
    {{run}} pytest -m physics {{args}}
