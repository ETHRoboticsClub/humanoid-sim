#!/usr/bin/env bash
# Run perf_bench.py over the lines of a matrix file: "<name> <args...>". Names starting with "master-" run against
# a copy of the workdir with the profile in $MASTER_PROFILE (for example master's ethrc_v1.yaml, saved beside it).
# Results: outputs/perf/<name>.json and .log. Skips names whose .json exists.
set -uo pipefail
export PATH="$HOME/.local/bin:$HOME/il-venv/bin:$PATH"
cd "$HOME/sky_workdir"
mkdir -p outputs/perf
MATRIX=${1:-spikes/perf/baseline.txt}
if [ -n "${MASTER_PROFILE:-}" ] && { [ ! -d "$HOME/sim-master" ] || [ "${REFRESH_MASTER:-0}" = 1 ]; }; then
  rm -rf "$HOME/sim-master" && mkdir -p "$HOME/sim-master"
  rsync -a --exclude outputs ./ "$HOME/sim-master/"
  cp "$MASTER_PROFILE" "$HOME/sim-master/profiles/ethrc_v1.yaml"
  (cd "$HOME/sim-master" && PYTHONPATH="$HOME/sim-master/src" humanoid-sim build-asset | tail -1)
fi
humanoid-sim build-asset | tail -1
while read -r name rest; do
  [ -z "$name" ] && continue; case "$name" in \#*) continue;; esac
  [ -f "outputs/perf/$name.json" ] && continue
  echo "== $(date +%H:%M:%S) $name $rest"
  if [[ "$name" == master-* ]] && [ -z "${MASTER_PROFILE:-}" ]; then echo "   skipped: no MASTER_PROFILE"; continue; fi
  if [[ "$name" == master-* ]]; then
    (cd "$HOME/sim-master" && PYTHONPATH="$HOME/sim-master/src" timeout 2400 python "$HOME/sky_workdir/spikes/perf/perf_bench.py" \
      $rest --out "$HOME/sky_workdir/outputs/perf/$name.json" > "$HOME/sky_workdir/outputs/perf/$name.log" 2>&1)
  else
    timeout 2400 python spikes/perf/perf_bench.py $rest --out "outputs/perf/$name.json" > "outputs/perf/$name.log" 2>&1
  fi
  rc=$?
  echo "   rc=$rc overflow=$(grep -c -i overflow "outputs/perf/$name.log") $(python -c "import json,sys;d=json.load(open('outputs/perf/$name.json'));print(round(d['env_steps_per_s']), round(d['ms_per_physics_step'],3), d['fraction_envs_up'], d['sync_error'])" 2>/dev/null)"
done < "$MATRIX"
echo "== done $(date +%H:%M:%S)"
