"""A minimal runtime client: attach to `humanoid-sim serve`, hold the robot at its baseline gains, wave the arms,
push the left forearm, then detach.

    humanoid-sim serve --segment /tmp/hsim/plant --device cuda:0 &
    python examples/runtime_client.py /tmp/hsim/plant --seconds 20
    # any robot in humanoid-urdf: pass --robot to both, e.g. --robot g1_29dof_with_hand_rev_1_0
    humanoid-sim view-runtime /tmp/hsim/plant          # in another shell, to watch

Controls' plant host does the same job in production: it writes one demand record per bus cycle and reads the
feedback back. Everything here goes through `humanoid_sim.runtime.abi.plant_abi`, which has no dependency beyond
NumPy, so a client doesn't need Isaac Lab.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from humanoid_sim.model.description import Description
from humanoid_sim.runtime.abi import plant_abi as abi

BUS_PERIOD_S = 0.0015  # Controls' EtherCAT cycle
WAVE = {"shoulder_flexion_l": -0.6, "shoulder_flexion_r": 0.6, "elbow_flexion_l": 0.5, "elbow_flexion_r": 0.5,
        "left_shoulder_pitch_joint": -0.6, "right_shoulder_pitch_joint": -0.6,
        "left_elbow_joint": 0.5, "right_elbow_joint": 0.5}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("segment")
    p.add_argument("--robot", default="ethrc_v1", help="the robot the server runs")
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--wave-hz", type=float, default=0.4)
    p.add_argument("--push-at", type=float, default=8.0, help="seconds after attaching (negative: no push)")
    args = p.parse_args(argv)

    desc = Description.load(args.robot)
    seg = abi.Segment.open(args.segment, robot_id=desc.robot, description_digest=desc.digest, wait_s=300)
    info = seg.descriptor()
    drives, bodies = info["drives"], info["bodies"]
    n = len(drives)
    # Push a forearm: on a suspended robot (serve's default support) the cable holds the base.
    pushed_body = next((b for b in bodies if "elbow" in b and (b.startswith("left") or b.endswith("l_1"))), bodies[0])
    joint_of = {d: desc.drives[d].joints[0] for d in drives}

    # Start from where the robot is, at the description's baseline gains (SI: N m/rad, N m s/rad).
    _, fb = seg.regions["feedback"].read()
    hold = fb["drive"]["position_rad"][:n].copy()
    demand = abi.pd_demand(hold, [desc.drives[d].baseline_stiffness for d in drives],
                           [desc.drives[d].baseline_damping for d in drives])
    # Drive frame = sign * joint angle; the description gives each direct drive's sign.
    wave = np.zeros(n)
    for i, d in enumerate(drives):
        if joint_of[d] in WAVE and desc.drives[d].crank is None:
            wave[i] = WAVE[joint_of[d]] * desc.drives[d].sign

    start, cycle, pushed = time.monotonic(), 0, False
    next_print = start
    while (now := time.monotonic()) - start < args.seconds:
        t = now - start
        ramp = min(1.0, t / 2.0)  # ease in over 2 s
        demand["target_position_rad"] = hold + wave * ramp * 0.5 * (1 - math.cos(2 * math.pi * args.wave_hz * t))
        cycle += 1
        seg.write_demands(demand, cycle=cycle)
        if not pushed and 0 <= args.push_at <= t:
            seg.send(abi.CMD_APPLY_DISTURBANCE, {"body": pushed_body, "force_n": [0.0, 40.0, 0.0],
                                                 "torque_nm": [0.0, 0.0, 0.0], "duration_s": 0.3})
            print(f"pushed {pushed_body}", flush=True)
            pushed = True
        if now >= next_print:
            try:
                got = seg.regions["feedback"].read()
            except abi.StaleRead:
                got = None
            if got is not None:
                f = got[1]
                print(f"t={t:5.1f}s sim={int(f['sim_time_ns']) / 1e9:7.2f}s late_steps={int(f['late_steps'])} "
                      f"max|torque|={np.abs(f['drive']['applied_torque_nm'][:n]).max():5.1f} Nm", flush=True)
            next_print = now + 1.0
        time.sleep(BUS_PERIOD_S)
    seq = seg.send(abi.CMD_DETACH, {})
    for _ in range(100):  # keep writing demands until the host acknowledges
        cycle += 1
        seg.write_demands(demand, cycle=cycle)
        try:
            result, message = seg.wait_ack(seq, timeout_s=0.02)
            print("detach:", result, message)
            break
        except TimeoutError:
            pass
    seg.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
