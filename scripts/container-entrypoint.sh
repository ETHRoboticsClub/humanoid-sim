#!/usr/bin/env bash
set -euo pipefail

# The image's installed package is the default. The dev profile deliberately
# replaces it with an editable install from a separate source mount.
case "${HUMANOID_SIM_DEV:-0}" in
  0) ;;
  1)
    test -f /workspace/humanoid-sim/pyproject.toml || {
      echo "HUMANOID_SIM_DEV=1 requires the repository at /workspace/humanoid-sim" >&2
      exit 2
    }
    uv pip install --python /opt/isaaclab-venv/bin/python --no-deps --editable /workspace/humanoid-sim
    ;;
  *) echo "HUMANOID_SIM_DEV must be 0 or 1" >&2; exit 2 ;;
esac

exec /opt/isaaclab-venv/bin/python -m humanoid_sim "$@"
