"""Compare recorded trajectories (perf_bench.py --record) against a reference.

    python traj_compare.py REF.npz OTHER.npz [OTHER2.npz ...]

Per file: joint and base-position differences to REF at 0.1, 0.5 and 1 s and over the whole record
(median and 90th percentile over envs), and the share of envs whose joints stay within 0.05 rad (max abs)
at 0.5 s. Compare a variant's numbers with those of a second baseline run (the GPU's run-to-run floor).
"""

import json
import sys

import numpy as np


def load(path):
    z = np.load(path)
    return z["state"], int(z["period_ns"]), len(z["joints"])


def main():
    ref, period_ns, nj = load(sys.argv[1])
    policy_dt = 0.02
    out = {}
    for path in sys.argv[2:]:
        other, _, _ = load(path)
        t = min(len(ref), len(other))
        dq = np.abs(other[:t, :, :nj] - ref[:t, :, :nj])          # [T, E, J]
        dp = np.linalg.norm(other[:t, :, nj:nj + 3] - ref[:t, :, nj:nj + 3], axis=-1)  # [T, E]
        row = {}
        for sec in (0.1, 0.5, 1.0):
            k = min(t - 1, round(sec / policy_dt))
            jmax = dq[k].max(-1)
            row[f"t{sec}s"] = {"joint_max_rad_median": float(np.median(jmax)), "joint_max_rad_p90": float(np.percentile(jmax, 90)),
                               "base_m_median": float(np.median(dp[k])), "base_m_p90": float(np.percentile(dp[k], 90)),
                               "envs_within_0.05rad": float((jmax < 0.05).mean())}
        row["all"] = {"joint_rms_rad": float(np.sqrt((dq ** 2).mean())), "base_rms_m": float(np.sqrt((dp ** 2).mean()))}
        out[path] = row
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
