#!/usr/bin/env bash
# One command from new excitation captures to the PACE holdout report:
#
#   sky/pace-launch.sh CAPTURE... [--description-root DIR] [--ankles cranks|hold] [--no-fit-ankles]
#                      [--generations N] [--num-envs N] [--bands N] [--window NAME:START:END ...]
#                      [--no-full-fit] [--dry] [--keep] [--name NAME] [--part PART] [--only PATTERN ...]
#                      [--parallel N] [--bound B] [--bias-bound-mrad M] [--delay-ms FROM TO] [--delay-start-ms D]
#                      [--ratio-max R] [--physics-tests]
#
# Stages the captures (and the description, when one is given: a humanoid-urdf checkout) in
# ~/ETHRC/runs/<date>-pace/input, launches sky/pace-capture.sky.yaml on cluster sim-pace (L40S first, autodown after
# 15 idle minutes), waits for the run, copies outputs/pace-run into ~/ETHRC/runs/<date>-pace/ and takes the cluster
# down. The node mirrors the same folder to s3://ethrc-rc-humanoid-ckpts/amoec/pace/<date>/ on its own, so the result
# survives this script being stopped. --dry: a tiny fit (2 generations, 16 envs) into <date>-pace-dry, with the
# replay and PACE physics tests first, to check the chain end to end. --keep: leave the cluster up (it still autodowns when idle).
# --name NAME: the run is <date>-NAME (default pace) locally and on S3; --part PART puts this launch in its PART
# subfolder on its own cluster (sim-NAME-PART), so several nodes can share one run, each with its own --only fits.
# --bound, --bias-bound-mrad, --delay-ms, --delay-start-ms, --ratio-max: the search box (fit-pace's options).
# --physics-tests: the replay and PACE physics tests first, as --dry runs them.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
date=$(TZ=Europe/Zurich date +%Y-%m-%d)
captures=() desc_root="" ankles=cranks fit_ankles=1 generations=40 num_envs=256 bands=3 windows="" full=1 dry=0 keep=0
physics_tests=0 name="" part="" only="" parallel=3 bound=10 bias=20 delay="0 20" delay_start="" ratio_max=1
while [ $# -gt 0 ]; do
  case $1 in
    --description-root) desc_root=$2; shift 2 ;;
    --ankles) ankles=$2; shift 2 ;;
    --no-fit-ankles) fit_ankles=0; shift ;;
    --generations) generations=$2; shift 2 ;;
    --num-envs) num_envs=$2; shift 2 ;;
    --bands) bands=$2; shift 2 ;;
    --window) windows="$windows $2"; shift 2 ;;
    --no-full-fit) full=0; shift ;;
    --dry) dry=1; shift ;;
    --keep) keep=1; shift ;;
    --physics-tests) physics_tests=1; shift ;;
    --name) name=$2; shift 2 ;;
    --part) part=$2; shift 2 ;;
    --only) only="$only $2"; shift 2 ;;
    --parallel) parallel=$2; shift 2 ;;
    --bound) bound=$2; shift 2 ;;
    --bias-bound-mrad) bias=$2; shift 2 ;;
    --delay-ms) delay="$2 $3"; shift 3 ;;
    --delay-start-ms) delay_start=$2; shift 2 ;;
    --ratio-max) ratio_max=$2; shift 2 ;;
    -h|--help) sed -n 2,22p "$0"; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) captures+=("$1"); shift ;;
  esac
done
[ ${#captures[@]} -gt 0 ] || { echo "no capture given" >&2; exit 2; }
topic=${name:-pace}
if [ "$dry" = 1 ]; then physics_tests=1; topic=$topic-dry; generations=${DRY_GENERATIONS:-2}; num_envs=${DRY_NUM_ENVS:-16}; fi
run="$HOME/ETHRC/runs/$date-$topic"
prefix="amoec/pace/$date${name:+-$name}"
[ "$dry" = 1 ] && prefix="$prefix-dry"
cluster=sim-${name:-pace}
if [ -n "$part" ]; then run="$run/$part"; prefix="$prefix/$part"; cluster="$cluster-$part"; fi
if [ -e "$run/input" ]; then
  echo "$run/input exists: a run of today is staged already; move it away or pass a different capture set" >&2
  exit 2
fi
mkdir -p "$run/input/captures"
for c in "${captures[@]}"; do
  c=${c%/}
  [ -f "$c/manifest.json" ] || { echo "$c: not an exc-* capture (no manifest.json)" >&2; exit 2; }
  rsync -a --exclude '*.zip' "$c/" "$run/input/captures/$(basename "$c")/"
done
desc_rev="" desc_dirty=""
if [ -n "$desc_root" ]; then
  mkdir -p "$run/input/description"
  rsync -a --exclude .git "$desc_root/ethrc_v1" "$run/input/description/"
  src_root=$desc_root
else
  src_root="$here/third_party/humanoid-urdf"
fi
desc_rev=$(git -C "$src_root" rev-parse HEAD 2>/dev/null || echo unknown)
desc_dirty=$(git -C "$src_root" status --porcelain 2>/dev/null | wc -l | tr -d ' ')
sim_rev=$(git -C "$here" rev-parse HEAD)
sim_dirty=$(git -C "$here" status --porcelain --untracked-files=no | wc -l | tr -d ' ')
python3 - "$run/input/inputs.json" "${captures[@]}" <<EOF
import json, sys
from pathlib import Path
caps = []
for c in sys.argv[2:]:
    m = json.loads((Path(c) / "manifest.json").read_text())
    caps.append({"path": str(Path(c).resolve()), "dataset_id": m["dataset_id"], "identity": m.get("identity", {}),
                 "sweep": (m.get("effective_action") or {}).get("sweep")})
json.dump({"captures": caps, "description_root": "${desc_root:-humanoid-sim third_party/humanoid-urdf}",
           "description_revision": "$desc_rev", "description_dirty_files": int("$desc_dirty"),
           "humanoid_sim_revision": "$sim_rev", "humanoid_sim_dirty_files": int("$sim_dirty"),
           "ankles": "$ankles", "fit_ankles": $fit_ankles, "generations": $generations, "num_envs": $num_envs,
           "bands": $bands, "windows": "$windows".split(), "full_fit": $full, "dry": $dry,
           "only": "$only".split(), "box": {"multiplier": $bound, "bias_mrad": $bias, "delay_ms": "$delay".split(),
                                           "delay_start_ms": "$delay_start", "friction_ratio_max": $ratio_max},
           "s3": "s3://ethrc-rc-humanoid-ckpts/$prefix"}, open(sys.argv[1], "w"), indent=1)
EOF
echo "[pace] staged in $run/input; humanoid-sim $sim_rev ($sim_dirty modified), description $desc_rev"
cd "$here"
sky launch -y -c "$cluster" -i 15 --down sky/pace-capture.sky.yaml \
  --env PACE_IN="$run/input" --env S3_PREFIX="$prefix" --env ANKLES="$ankles" --env FIT_ANKLES="$fit_ankles" \
  --env GENERATIONS="$generations" --env NUM_ENVS="$num_envs" --env BANDS="$bands" --env WINDOWS="$windows" \
  --env FULL_FIT="$full" --env PHYSICS_TESTS="$physics_tests" --env ONLY="${only# }" --env PARALLEL="$parallel" \
  --env BOUND="$bound" --env BIAS_BOUND_MRAD="$bias" --env DELAY_MS="$delay" --env DELAY_START_MS="$delay_start" \
  --env RATIO_MAX="$ratio_max" 2>&1 | tee "$run/launch.log" || true
mkdir -p "$run/out"
rsync -a "$cluster:sky_workdir/outputs/pace-run/" "$run/out/" || echo "[pace] could not copy from the node; the result is in s3://ethrc-rc-humanoid-ckpts/$prefix"
if [ "$keep" = 0 ]; then
  sky down -y "$cluster" && sky status "$cluster" 2>&1 | tail -3
fi
echo "[pace] $(cat "$run/out/STATUS" 2>/dev/null || echo 'no STATUS'): $run/out/report/pace-report.md"
