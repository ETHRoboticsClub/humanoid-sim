# syntax=docker/dockerfile:1
# Kitless recipe follows Isaac Lab's Dockerfile.kitless at the pinned commit.
# https://github.com/isaac-sim/IsaacLab/blob/b9a673a29f3a7f09508f339c3ec370fdda2f542f/docker/Dockerfile.kitless
ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE} AS builder
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG TARGETARCH
ARG ISAACLAB_COMMIT=b9a673a29f3a7f09508f339c3ec370fdda2f542f
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 \
    ISAACLAB_PATH=/opt/isaaclab VIRTUAL_ENV=/opt/isaaclab-venv \
    UV_PROJECT_ENVIRONMENT=/opt/isaaclab-venv UV_PYTHON=/usr/bin/python3.12 \
    UV_PYTHON_PREFERENCE=only-system
ENV PATH=/opt/isaaclab-venv/bin:/usr/local/bin:${PATH}
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake git libglib2.0-0 python3 python3-dev \
    && rm -rf /var/lib/apt/lists/*
# Select the binary for the target, including when building through emulation.
COPY scripts/install-uv.py /tmp/install-uv.py
RUN python3 /tmp/install-uv.py "${TARGETARCH}"
RUN uv --version
RUN git init "${ISAACLAB_PATH}" \
    && git -C "${ISAACLAB_PATH}" remote add origin https://github.com/isaac-sim/IsaacLab.git \
    && git -C "${ISAACLAB_PATH}" fetch --depth 1 origin "${ISAACLAB_COMMIT}" \
    && git -C "${ISAACLAB_PATH}" checkout --detach FETCH_HEAD \
    && test "$(git -C "${ISAACLAB_PATH}" rev-parse HEAD)" = "${ISAACLAB_COMMIT}"
WORKDIR /opt/isaaclab
# Frozen upstream resolution; only the Viser extra is enabled. Upstream's base
# graph includes RL utilities, but this project installs and runs no trainer.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --seed "${VIRTUAL_ENV}" \
    && uv sync --frozen --inexact --extra viser \
    && python -c "import torch, yaml, newton, isaaclab_newton, viser" \
    && test ! -e /opt/isaaclab/_isaac_sim \
    && rm -rf /opt/isaaclab/.git
COPY pyproject.toml README.md runtime.lock.yaml /opt/humanoid-sim/
COPY src/ /opt/humanoid-sim/src/
COPY profiles/ /opt/humanoid-sim/profiles/
RUN uv pip install --python "${VIRTUAL_ENV}/bin/python" --no-deps /opt/humanoid-sim

FROM ${BASE_IMAGE} AS runtime
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG ISAACLAB_COMMIT=b9a673a29f3a7f09508f339c3ec370fdda2f542f
LABEL org.opencontainers.image.description="Humanoid simulation with Isaac Lab and Newton (kitless)" \
    org.ethrc.isaaclab.commit=${ISAACLAB_COMMIT}
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 \
    ISAACLAB_PATH=/opt/isaaclab VIRTUAL_ENV=/opt/isaaclab-venv \
    UV_PROJECT_ENVIRONMENT=/opt/isaaclab-venv UV_PYTHON=/usr/bin/python3.12 \
    UV_PYTHON_PREFERENCE=only-system UV_NO_SYNC=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    HOME=/home/isaaclab WARP_CACHE_PATH=/home/isaaclab/.cache/warp \
    HUMANOID_SIM_DEV=0
ENV PATH=/opt/isaaclab-venv/bin:/usr/local/bin:${PATH}
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libegl1 libgl1 libglib2.0-0 libgomp1 libopengl0 \
    libpython3.12 libvulkan1 libx11-6 libxext6 libxrender1 python3 \
    && rm -rf /var/lib/apt/lists/* \
    && if getent passwd ubuntu >/dev/null; then userdel -r ubuntu; fi \
    && if getent group ubuntu >/dev/null; then groupdel ubuntu; fi \
    && groupadd --gid 1000 isaaclab \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash isaaclab \
    && mkdir -p /home/isaaclab/.cache/warp /home/isaaclab/.cache/uv /runs /workspace/humanoid-sim \
    && chown -R 1000:1000 /home/isaaclab /runs /workspace/humanoid-sim
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uvx /usr/local/bin/
COPY --from=builder --chown=1000:1000 /opt/isaaclab-venv /opt/isaaclab-venv
COPY --from=builder /opt/isaaclab /opt/isaaclab
COPY --from=builder /opt/humanoid-sim /opt/humanoid-sim
COPY --chmod=755 scripts/container-entrypoint.sh /usr/local/bin/humanoid-sim-entrypoint
USER isaaclab
WORKDIR /runs
ENTRYPOINT ["/usr/local/bin/humanoid-sim-entrypoint"]
CMD ["smoke", "--backend", "isaac", "--headless", "--steps", "120", "--output-dir", "/runs/smoke"]
