#!/usr/bin/env bash
# Shared GPU-node setup: Isaac Lab release/3.0.0 at the pinned commit, resolved from its own uv.lock.
set -euo pipefail
nvidia-smi --query-gpu=name,driver_version --format=csv
sudo apt-get update -q && sudo apt-get install -yq --no-install-recommends git libgl1 libegl1 libglib2.0-0 \
  libgomp1 libsm6 libxt6 libxrender1 libxext6 > /dev/null
curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
export PATH="$HOME/.local/bin:$PATH"
if [ ! -d "$HOME/isaaclab/.git" ]; then
  git init -q "$HOME/isaaclab" && git -C "$HOME/isaaclab" remote add origin https://github.com/isaac-sim/IsaacLab.git
fi
git -C "$HOME/isaaclab" fetch -q --depth 1 origin "$ISAACLAB_COMMIT" && git -C "$HOME/isaaclab" checkout -q --detach FETCH_HEAD
cd "$HOME/isaaclab"
uv python install 3.12
export UV_PROJECT_ENVIRONMENT="$HOME/il-venv"
uv venv -q --python 3.12 --seed "$UV_PROJECT_ENVIRONMENT" --allow-existing
uv sync -q --frozen --inexact --extra importers --extra rerun --extra viser
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
# torch cu130 needs driver >= 580; fall back to the same torch built for CUDA 12.8 on older drivers.
if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "[setup] torch cu130 cannot use this driver; installing torch 2.11.0+cu128"
  uv pip install -q --python "$PY" --reinstall torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu128
fi
"$PY" -c "import torch, warp as wp; wp.init(); print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'warp devices', wp.get_cuda_device_count())"
uv pip install -q --python "$PY" --no-deps -e "$HOME/sky_workdir"
