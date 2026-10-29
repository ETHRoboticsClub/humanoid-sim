#!/usr/bin/env bash
# The PACE run of sky/pace-capture.sky.yaml, on the node: every capture under $PACE_IN/captures checked, converted
# from the description it was recorded on when its joint names are not this one's, cut into windows, fitted on each
# window with the others held out (and once on all of them), and reported. Writes outputs/pace-run/ and mirrors it
# to /ckpt/$S3_PREFIX (the club bucket). Exit status: 0 when the report was written.
set -uo pipefail
export PATH="$HOME/.local/bin:$HOME/il-venv/bin:$PATH"
cd "$HOME/sky_workdir"
out=outputs/pace-run
mkdir -p "$out"
exec > >(tee -a "$out/run.log") 2>&1
status=FAILED
finish() {
  echo "$status $(TZ=Europe/Zurich date -Iseconds)" > "$out/STATUS"
  if [ -n "${S3_PREFIX:-}" ] && [ -d /ckpt ]; then
    mkdir -p "/ckpt/$S3_PREFIX"
    for _ in 1 2 3 4 5; do
      rsync -a --no-perms --no-owner --no-group --omit-dir-times "$out/" "/ckpt/$S3_PREFIX/" && break
      sleep 5
    done && echo "[pace] mirrored to s3://ethrc-rc-humanoid-ckpts/$S3_PREFIX"
  fi
}
trap finish EXIT
step() { echo; echo "[pace] $(TZ=Europe/Zurich date +%H:%M:%S) $*"; }

if [ -d "$PACE_IN/description" ]; then
  export HUMANOID_SIM_DESCRIPTION_ROOT="$PACE_IN/description"
fi
cp "$PACE_IN/inputs.json" "$out/inputs.json" 2>/dev/null || true
nvidia-smi --query-gpu=name --format=csv,noheader | tee "$out/gpu.txt"

step "asset"
humanoid-sim --robot "$ROBOT" build-asset | tail -1 | tee "$out/asset.txt" || exit 1

if [ "${PHYSICS_TESTS:-0}" = 1 ]; then
  step "physics tests: replay and PACE"
  uv pip install -q --python "$HOME/il-venv/bin/python" "pytest>=9,<10"
  python -m pytest -m physics -k "pace or replay or gains" -q -p no:cacheprovider > "$out/physics-tests.log" 2>&1
  tail -3 "$out/physics-tests.log"
fi

# The drives the fit leaves out: the ankle cranks unless FIT_ANKLES=1.
ankles=$(python -c "
from humanoid_sim.model.description import Description
d = Description.load('$ROBOT')
print(' '.join(n for lk in d.linkages for n in (lk.upper, lk.lower)))")
skip=()
[ "$FIT_ANKLES" = 1 ] || skip=(--skip-drives $ankles)
stamps=()
[ "$FRAME_STAMPS" = 1 ] && stamps=(--frame-stamps)

step "convert ($ANKLES ankles)"
mkdir -p "$out/captures"
for cap in "$PACE_IN"/captures/exc-*; do
  name=$(basename "$cap")
  if python -c "
import sys; from pathlib import Path
from humanoid_sim.model.description import Description
from humanoid_sim.replay.convert import is_legacy
sys.exit(0 if is_legacy(Path('$cap'), Description.load('$ROBOT')) else 1)"; then
    humanoid-sim --robot "$ROBOT" convert-capture "$cap" --out "$out/captures" --from "$LEGACY" \
      --ankles "$ANKLES" || exit 1
  else
    echo "$name: already in $ROBOT's joint names"
    cp -R "$cap" "$out/captures/$name"
  fi
done

step "check-bag"
humanoid-sim --robot "$ROBOT" check-bag "$out"/captures/exc-* --base fixed "${stamps[@]}" "${skip[@]}" \
  | tee "$out/check-bag.txt" || { echo "[pace] a capture fails the contract"; exit 1; }

step "windows"
win=()
for w in $WINDOWS; do win+=(--window "$w"); done
humanoid-sim segment-capture "$out"/captures/exc-* --out "$out/windows" --bands "$BANDS" "${win[@]}" \
  | tee "$out/windows.jsonl" || exit 1
mapfile -t windows < <(ls -d "$out"/windows/exc-*)
[ "${#windows[@]}" -ge 2 ] || { echo "[pace] fewer than two windows: nothing to hold out"; exit 1; }

common=(--base fixed --num-envs "$NUM_ENVS" --generations "$GENERATIONS" --device cuda:0 "${stamps[@]}" "${skip[@]}")

step "self-collision: the first window replayed at the description's values, with and without"
mkdir -p "$out/selfcol"
humanoid-sim --robot "$ROBOT" replay "${windows[0]}" --base fixed --out "$out/selfcol" > "$out/selfcol/on.log" 2>&1 &
p_on=$!
humanoid-sim --robot "$ROBOT" replay "${windows[0]}" --base fixed --no-self-collision --out "$out/selfcol" \
  > "$out/selfcol/off.log" 2>&1 &
p_off=$!

step "fits: each window with the others held out$([ "$FULL_FIT" = 1 ] && echo ', and all windows together')"
mkdir -p "$out/fits"
jobs_=()
for w in "${windows[@]}"; do
  held=()
  for o in "${windows[@]}"; do [ "$o" = "$w" ] || held+=("$o"); done
  jobs_+=("fit-$(basename "$w" | sed 's/^exc-\([0-9a-f]\{8\}\)[0-9a-f]*-/\1-/')|$w --holdout ${held[*]}")
done
[ "$FULL_FIT" = 1 ] && jobs_+=("fit-all|${windows[*]}")
running=0
for job in "${jobs_[@]}"; do
  name=${job%%|*}
  # shellcheck disable=SC2086
  (humanoid-sim --robot "$ROBOT" fit-pace ${job#*|} "${common[@]}" --out "$out/fits/$name" \
     > "$out/fits/$name.log" 2>&1; echo "$name exit $?" >> "$out/fits/exits.txt") &
  running=$((running + 1))
  if [ "$running" -ge "$PARALLEL" ]; then wait -n; running=$((running - 1)); fi
done
wait
cat "$out/fits/exits.txt"
wait "$p_on" "$p_off" 2>/dev/null
for f in on off; do echo "selfcol $f: $(tail -c 2000 "$out/selfcol/$f.log" | grep -E '^\{' | tail -1)"; done
python - "$out/selfcol" <<'EOF' | tee "$out/selfcol/summary.txt"
import json, sys
from pathlib import Path
for s in sorted(Path(sys.argv[1]).glob("*/summary.json")):
    d = json.loads(s.read_text())
    rms = d["rms_position_rad"]
    vals = [v for v in rms.values() if v == v]
    print(f"{s.parent.name}: self_collision={d.get('self_collision')} mean position RMS {1e3 * sum(vals) / max(len(vals), 1):.2f} mrad, "
          f"worst {max(rms, key=lambda k: rms[k] if rms[k] == rms[k] else 1e9)} "
          f"{1e3 * max((v if v == v else float('inf')) for v in rms.values()):.2f} mrad, non-finite {len(rms) - len(vals)}")
EOF

step "report"
fits=$(for d in "$out"/fits/fit-*/; do [ -f "$d/fit-pace.json" ] && echo "${d%/}"; done)
[ -n "$fits" ] || { echo "[pace] no fit finished"; exit 1; }
# shellcheck disable=SC2086
humanoid-sim --robot "$ROBOT" pace-report $fits --out "$out/report" || exit 1
status=DONE
