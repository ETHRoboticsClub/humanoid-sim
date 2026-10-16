#!/usr/bin/env bash
set -euo pipefail
# With the repository mounted at /workspace/humanoid-sim, run it editable instead of the baked copy.
if [[ -f /workspace/humanoid-sim/pyproject.toml ]]; then
  uv pip install -q --python /opt/isaaclab-venv/bin/python --no-deps --editable /workspace/humanoid-sim
  export HUMANOID_SIM_DESCRIPTION_ROOT=${HUMANOID_SIM_DESCRIPTION_ROOT:-/workspace/humanoid-sim/third_party/humanoid-urdf}
fi
if [[ "${1:-}" == "python" || "${1:-}" == "pytest" || "${1:-}" == "bash" ]]; then
  exec "$@"
fi
exec /opt/isaaclab-venv/bin/python -m humanoid_sim "$@"
