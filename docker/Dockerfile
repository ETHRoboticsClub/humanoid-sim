# Isaac Lab 3.0 (kitless, Newton) + humanoid-sim. Multi-arch: linux/amd64 (GPU hosts) and linux/arm64
# (Mac development and tests, DGX Spark). Isaac Lab is resolved from its own frozen uv.lock at
# the pinned commit; humanoid-sim installs into the same environment without touching that resolution.
ARG BASE_IMAGE=public.ecr.aws/ubuntu/ubuntu:24.04
FROM ${BASE_IMAGE}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG ISAACLAB_COMMIT=32712682b69edd488eea26689da6905c48d31037
ARG ISAACLAB_EXTRAS="importers rerun viser"
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 \
    ISAACLAB_PATH=/opt/isaaclab VIRTUAL_ENV=/opt/isaaclab-venv \
    UV_PROJECT_ENVIRONMENT=/opt/isaaclab-venv UV_PYTHON=/usr/bin/python3.12 UV_PYTHON_PREFERENCE=only-system \
    UV_LINK_MODE=copy PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV PATH=/opt/isaaclab-venv/bin:${PATH}
LABEL org.ethrc.isaaclab.commit=${ISAACLAB_COMMIT}

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake git git-lfs libegl1 libgl1 libglib2.0-0 libgomp1 libopengl0 \
      libsm6 libvulkan1 libx11-6 libxext6 libxrender1 libxt6 python3.12 python3.12-dev python3.12-venv \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /usr/local/bin/

RUN git init -q "${ISAACLAB_PATH}" \
    && git -C "${ISAACLAB_PATH}" remote add origin https://github.com/isaac-sim/IsaacLab.git \
    && git -C "${ISAACLAB_PATH}" fetch -q --depth 1 origin "${ISAACLAB_COMMIT}" \
    && git -C "${ISAACLAB_PATH}" checkout -q --detach FETCH_HEAD \
    && test "$(git -C "${ISAACLAB_PATH}" rev-parse HEAD)" = "${ISAACLAB_COMMIT}"
WORKDIR /opt/isaaclab
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv -q --seed "${VIRTUAL_ENV}" \
    && uv sync --frozen --inexact $(printf -- '--extra %s ' ${ISAACLAB_EXTRAS}) \
    && python -c "import torch, warp, newton, mujoco_warp, isaaclab, isaaclab_newton, rerun, pxr" \
    && rm -rf /opt/isaaclab/.git

# Preserve full constraint capacity while scheduling only active rows in MJWarp's
# constraint update and sparse force assembly; skip unused tactile scratch and
# unobservable intermediate RK4 acceleration diagnostics.
# Fail the build if the pin drifts.
COPY docker/apply_mjwarp_patch.py docker/verify_mjwarp_active_rows.py docker/verify_mjwarp_sensor_guard.py docker/verify_mjwarp_rk4_stages.py /opt/humanoid-sim-build/
COPY docker/patches/ /opt/humanoid-sim-build/patches/
RUN python /opt/humanoid-sim-build/apply_mjwarp_patch.py
# Keep converged PADMM worlds out of subsequent LLTB solves without changing
# the equations, stopping criteria, or the converged solution.
COPY docker/apply_kamino_patch.py /opt/humanoid-sim-build/
RUN python /opt/humanoid-sim-build/apply_kamino_patch.py
LABEL org.ethrc.mjwarp.solver.sha256=c737461c682ec74fcd9fcd4714f63b063d22bbac1f1c1e71470e1249d11e355f
LABEL org.ethrc.mjwarp.sensor.sha256=1b64d41dac172b7ba6488bfe69a22dfc90363907d7fd9c971a1123f5a02d5eb6
LABEL org.ethrc.mjwarp.forward.sha256=6f4220103cbb0605bb9f1201d22da9e0c2f3fb3ac985fa5c4666e77f24acdc14

# humanoid-sim: dependencies already satisfied by the Isaac Lab environment (torch, numpy, PyYAML).
# The dev entrypoint swaps this copy for an editable install from /workspace/humanoid-sim.
COPY pyproject.toml README.md /opt/humanoid-sim/
COPY src/ /opt/humanoid-sim/src/
RUN uv pip install -q --python "${VIRTUAL_ENV}/bin/python" --no-deps /opt/humanoid-sim \
    && uv pip install -q --python "${VIRTUAL_ENV}/bin/python" "pytest>=9,<10" "rosbags>=0.11,<0.12"
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/humanoid-sim-entrypoint
ENV WARP_CACHE_PATH=/cache/warp
WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/humanoid-sim-entrypoint"]
CMD ["--help"]
