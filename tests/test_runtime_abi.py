"""Runtime segment ABI: the C header matches the Python layout, the handshake checks, and the seqlock never
hands out a torn record."""

import multiprocessing as mp
import shutil
import subprocess

import numpy as np
import pytest

from humanoid_sim.runtime.abi import plant_abi as abi

HEADER_H = abi.Path(abi.__file__).with_name("plant_abi.h")

STRUCTS = {"HsimHeader": abi.HEADER, "HsimDriveDemand": abi.DRIVE_DEMAND, "HsimDriveFeedback": abi.DRIVE_FEEDBACK,
           "HsimImuSample": abi.IMU_SAMPLE, "HsimSole": abi.SOLE, "HsimDescriptor": abi.DESCRIPTOR,
           "HsimDemandRecord": abi.DEMAND_RECORD,
           "HsimDemands": abi.DEMANDS, "HsimFeedback": abi.FEEDBACK, "HsimMailbox": abi.MAILBOX, "HsimAck": abi.ACK,
           "HsimTruth": abi.TRUTH}


def _create(path, **kw):
    args = dict(robot_id="ethrc_v1", description_digest="d" * 64, sim_revision="test", num_drives=22,
                num_joints=22, physics_period_ns=4_000_000, bus_period_ns=1_500_000,
                descriptor={"drives": [f"d{i}" for i in range(22)]})
    return abi.Segment.create(path, **(args | kw))


def test_c_header_has_the_python_layout(tmp_path):
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler")
    lines = []
    for struct, dtype in STRUCTS.items():
        lines.append(f'printf("{struct} %zu\\n", sizeof({struct}));')
        for name in dtype.names:
            lines.append(f'printf("{struct}.{name} %zu\\n", offsetof({struct}, {name}));')
    src = tmp_path / "layout.c"
    src.write_text(f'#include <stdio.h>\n#include "{HEADER_H}"\nint main(void) {{\n' + "\n".join(lines) +
                   '\nprintf("hash %llu\\n", (unsigned long long)HSIM_PLANT_LAYOUT_HASH);\n'
                   'printf("bytes %d\\n", HSIM_PLANT_SEGMENT_BYTES);\n'
                   + "".join(f'printf("offset.{n} %d\\n", HSIM_OFFSET_{n.upper()});\n' for n, _ in abi.REGIONS)
                   + "return 0;\n}\n")
    exe = tmp_path / "layout"
    subprocess.run([cc, "-std=c11", "-Wall", "-Werror", str(src), "-o", str(exe)], check=True)
    got = dict(line.rsplit(" ", 1) for line in subprocess.run([str(exe)], capture_output=True, text=True,
                                                               check=True).stdout.splitlines())
    want = {}
    for struct, dtype in STRUCTS.items():
        want[struct] = dtype.itemsize
        for name in dtype.names:
            want[f"{struct}.{name}"] = dtype.fields[name][1]
    want["hash"] = abi.LAYOUT_HASH
    want["bytes"] = abi.SEGMENT_BYTES
    want |= {f"offset.{n}": abi.OFFSETS[n] for n, _ in abi.REGIONS}
    assert {k: int(v) for k, v in got.items()} == want


def test_the_gantry_straps_are_appended_to_the_truth():
    """ABI 3.4 added the straps after the 3.3 fields, so a 3.3 reader's offsets still hold."""
    assert abi.TRUTH.names[-3:] == ("gantry_command_m", "gantry_strap_top_m", "gantry_strap_bottom_m")
    for name in ("gantry_strap_top_m", "gantry_strap_bottom_m"):
        assert abi.TRUTH.fields[name][0].shape == (abi.MAX_STRAPS, 3)


def test_open_checks_identity_and_layout(tmp_path):
    path = tmp_path / "plant"
    seg = _create(path)
    client = abi.Segment.open(path, robot_id="ethrc_v1", description_digest="d" * 64)
    assert client.session_id == seg.session_id and client.num_drives == 22
    assert client.descriptor()["drives"][3] == "d3"
    with pytest.raises(abi.AbiError, match="robot id"):
        abi.Segment.open(path, robot_id="g1")
    with pytest.raises(abi.AbiError, match="digest"):
        abi.Segment.open(path, description_digest="e" * 64)
    seg.header["layout_hash"] = 1
    with pytest.raises(abi.AbiError, match="layout hash"):
        abi.Segment.open(path)
    seg.header["layout_hash"] = abi.LAYOUT_HASH
    seg.end()
    with pytest.raises(abi.AbiError, match="not ready"):
        abi.Segment.open(path)
    with pytest.raises(FileNotFoundError):
        abi.Segment.open(tmp_path / "missing")


def test_region_rejects_torn_and_corrupt_copies(tmp_path):
    seg = _create(tmp_path / "plant")
    fb = seg.regions["feedback"]
    assert fb.read() is None
    rec = np.zeros((), abi.FEEDBACK)
    rec["step"], rec["drive"]["position_rad"][:3] = 7, [0.1, 0.2, 0.3]
    s = fb.write(rec)
    got_seq, got = fb.read()
    assert got_seq == s and int(got["step"]) == 7 and got["drive"]["position_rad"][2] == 0.3
    fb._seq[0] = s + 1  # mid-write
    with pytest.raises(abi.StaleRead):
        fb.read(attempts=10)
    fb._seq[0] = s
    fb._body[40] ^= 0xFF  # payload changed without a new CRC
    with pytest.raises(abi.StaleRead):
        fb.read(attempts=10)


def test_mailbox_round_trip(tmp_path):
    path = tmp_path / "plant"
    sim = _create(path)
    client = abi.Segment.open(path)
    seq = client.send(abi.CMD_SET_SUPPORT, {"support": "free"})
    got = sim.poll_command(0)
    assert got == (seq, abi.CMD_SET_SUPPORT, {"support": "free"})
    assert sim.poll_command(seq) is None
    sim.ack(seq, abi.ACK_REJECTED, "no")
    assert client.wait_ack(seq, timeout_s=1.0) == (abi.ACK_REJECTED, "no")


def test_demands_ring_round_trip(tmp_path):
    path = tmp_path / "plant"
    sim = _create(path)
    client = abi.Segment.open(path)
    n = 22
    demand = abi.pd_demand(np.linspace(-1, 1, n), 320.0, 12.0, torque_limit_nm=np.full(n, 40.0))
    demand["law"][0] = abi.LAW_FREEZE
    client.write_demands(demand, cycle=5, sent_mono_ns=123)
    _, rec = sim.regions["demands"].read()
    assert int(rec["count"]) == 1 and int(rec["session_id"]) == sim.session_id and int(rec["num_drives"]) == n
    r = rec["ring"][0]
    assert int(r["cycle"]) == 5 and int(r["sent_mono_ns"]) == 123
    np.testing.assert_array_equal(r["drive"]["target_position_rad"][:n], demand["target_position_rad"])
    assert int(r["drive"]["law"][0]) == abi.LAW_FREEZE and int(r["drive"]["law"][1]) == abi.LAW_PD
    assert float(r["drive"]["stiffness_nm_per_rad"][n - 1]) == 320.0
    assert float(r["drive"]["torque_limit_nm"][n - 1]) == 40.0


def test_the_demand_ring_keeps_the_last_cycles_and_survives_a_reopen(tmp_path):
    path = tmp_path / "plant"
    sim = _create(path)
    client = abi.Segment.open(path)
    for cycle in range(1, abi.DEMAND_RING + 4):
        client.write_demands(abi.pd_demand(np.full(22, float(cycle)), 1.0, 0.1), cycle=cycle, sent_mono_ns=cycle)
    client.close()
    again = abi.Segment.open(path)  # a client reopening the session's segment continues the ring
    again.write_demands(abi.pd_demand(np.zeros(22), 1.0, 0.1), cycle=99)
    _, rec = sim.regions["demands"].read()
    count = int(rec["count"])
    assert count == abi.DEMAND_RING + 4
    kept = [int(rec["ring"][k % abi.DEMAND_RING]["cycle"]) for k in range(count - abi.DEMAND_RING, count)]
    assert kept == list(range(5, abi.DEMAND_RING + 4)) + [99]


def test_a_pd_demand_has_no_torque_limit():
    demand = abi.pd_demand(np.zeros(3), [1.0, 2.0, 3.0], 0.5)
    assert (demand["law"] == abi.LAW_PD).all() and np.isinf(demand["torque_limit_nm"]).all()
    assert demand["stiffness_nm_per_rad"].tolist() == [1.0, 2.0, 3.0] and demand["damping_nm_s_per_rad"][2] == 0.5


def _writer(path, n, ready):
    seg = abi.Segment.open(path)
    fb = seg.regions["feedback"]
    rec = np.zeros((), abi.FEEDBACK)
    ready.set()
    for i in range(1, n + 1):
        rec["step"] = i
        rec["drive"]["position_rad"][:] = i
        rec["imu"]["seq"][:] = i
        fb.write(rec)


def test_concurrent_reads_are_never_torn(tmp_path):
    path = tmp_path / "plant"
    seg = _create(path)
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    p = ctx.Process(target=_writer, args=(path, 20000, ready))
    p.start()
    ready.wait(30)
    fb, seen = seg.regions["feedback"], 0
    while p.is_alive() or seen < 20000:
        try:
            got = fb.read()
        except abi.StaleRead:  # the writer was descheduled mid-write: not a torn copy
            continue
        if got is None:
            continue
        rec = got[1]
        step = int(rec["step"])
        assert (rec["drive"]["position_rad"] == step).all() and (rec["imu"]["seq"] == step).all()
        seen = max(seen, step)
        if not p.is_alive() and seen < 20000:
            break
    p.join()
    assert p.exitcode == 0 and seen == 20000
