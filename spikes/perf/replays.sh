#!/usr/bin/env bash
# Replays of the robot captures in $CAPS for each "solver:period_ms:integrator" in $RUNS, one process per capture.
# OUT (default outputs/perf/replay); extra env PROFILE_TAG names the code/profile variant in the output path.
set -uo pipefail
export PATH="$HOME/.local/bin:$HOME/il-venv/bin:$PATH"
cd "$HOME/sky_workdir"
OUT=${OUT:-outputs/perf/replay}${PROFILE_TAG:+-$PROFILE_TAG}
mkdir -p "$OUT"
humanoid-sim build-asset --base-joints | tail -1
for cap in ${CAPS:-/data/exc-*}; do
  (
    for r in $RUNS; do
      IFS=: read -r solver period integ <<< "$r"
      humanoid-sim --solver "$solver" replay "$cap" --device cuda:0 --period-ms "$period" ${integ:+--integrator "$integ"} \
        --out "$OUT" > "$OUT/$(basename "$cap")_${solver}_${period}_${integ}.log" 2>&1 || echo "FAILED $cap $r"
    done
  ) &
done
wait
python3 - "$OUT" <<'PY'
import json, sys, pathlib, statistics
for s in sorted(pathlib.Path(sys.argv[1]).glob("*/summary.json")):
    d = json.loads(s.read_text())
    rms = d["rms_position_rad"]
    vals = list(rms.values()) if isinstance(rms, dict) else [rms]
    print(f"{s.parent.name:90s} mean_rms_pos={statistics.mean(vals):.5f} rad  wall={d['wall_s']:.0f}s sim={d['sim_s']:.1f}s")
PY
