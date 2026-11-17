"""Kamino's dense operator capacity at training scale: the V1 with joint give on the joints the 2026-11-16 fit measured
builds and steps at 4096 envs, and every V1 asset variant's largest world fits the dense operator's int32 indexing at
4096 envs with headroom left (``contact_capacity.kamino_constraint_rows``).

Kamino sizes each world's dense block by its largest possible constraint count (joint rows, bounded effort rows,
limits, three rows per contact of ``max_contacts_per_world``) and indexes all worlds' blocks with int32 offsets. Before
the profile's budget fitted it, the give asset needed 730 rows a world, 4096 x 730^2 > 2^31, and failed at build with a
negative array shape. A new mechanism or closed chain (the V2) adds rows: these checks name the row count it reaches.

Marked ``physics``: CUDA, in the Isaac Lab image with the assets built."""

import math

import pytest
import torch
from conftest import built_usd

pytestmark = pytest.mark.physics

TRAIN_ENVS = 4096
GIVE_MEASURED = ("hip_flexion_l", "hip_flexion_r", "hip_abduction_l", "hip_abduction_r", "knee_flexion_l",
                 "knee_flexion_r")
GIVE_GAINS = {"stiffness_nm_per_rad": 700.0, "damping_nm_s_per_rad": 1.5}
COMPLIANCE_GAINS = {"stiffness_nm_per_rad": 600.0, "damping_nm_s_per_rad": 0.1}


def _cuda():
    pytest.importorskip("isaaclab_newton")
    if not torch.cuda.is_available():
        pytest.skip("physics checks run on CUDA")


def _scene(num_envs, usd, overrides):
    import isaaclab.sim as sim_utils
    from isaaclab.utils import instantiate

    from humanoid_sim.train.scene import robot_build, robot_scene_cfg, robot_sim_cfg

    sim = sim_utils.SimulationContext(robot_sim_cfg("ethrc_v1", device="cuda:0", solver="kamino",
                                                    physics_overrides=overrides))
    with robot_build("ethrc_v1", solver="kamino", physics_overrides=overrides):
        scene = instantiate(robot_scene_cfg(num_envs, usd, robot="ethrc_v1", pose="home",
                                            physics_overrides=overrides))
        sim.reset()
    return sim, scene


def _variant(name):
    if name == "rigid":
        return built_usd("ethrc_v1"), {}
    if name == "ankle_compliance":
        return built_usd("ethrc_v1", ankle_compliance=True), {"ankle_compliance": COMPLIANCE_GAINS}
    if name == "give_measured":
        return (built_usd("ethrc_v1", joint_give=GIVE_MEASURED),
                {"joint_give": {**GIVE_GAINS, "stiffness_nm_per_rad": {j: 700.0 for j in GIVE_MEASURED}}})
    if name == "give_all":  # every candidate joint: the most rows a V1 asset variant has today
        from humanoid_sim.model.description import Description
        from humanoid_sim.transmissions.compliance import give_candidates

        joints = tuple(give_candidates(Description.load("ethrc_v1")))
        return (built_usd("ethrc_v1", joint_give=joints),
                {"joint_give": {**GIVE_GAINS, "stiffness_nm_per_rad": {j: 700.0 for j in joints}}})
    raise ValueError(name)


@pytest.mark.parametrize("variant", ["rigid", "ankle_compliance", "give_measured", "give_all"])
def test_every_v1_variant_fits_the_dense_operator_at_training_scale(variant):
    _cuda()
    import isaaclab.sim as sim_utils
    from isaaclab_newton.physics import NewtonManager

    from humanoid_sim.physics.contact_capacity import kamino_constraint_rows

    usd, overrides = _variant(variant)
    _scene(2, usd, overrides)
    try:
        rows = kamino_constraint_rows(NewtonManager._solver)
    finally:
        sim_utils.SimulationContext.clear_instance()
    assert rows["max_worlds"] >= TRAIN_ENVS, f"{variant}: {rows}"


def test_the_v1_with_measured_joint_give_builds_and_steps_at_4096_envs():
    _cuda()
    import isaaclab.sim as sim_utils

    usd, overrides = _variant("give_measured")
    sim, scene = _scene(TRAIN_ENVS, usd, overrides)
    try:
        robot = scene["robot"]
        for _ in range(20):
            scene.write_data_to_sim()
            sim.step()
            scene.update(0.001)  # the V1 profile's physics period
        q = robot.data.joint_pos.torch
        assert torch.isfinite(q).all()
        height = robot.data.root_link_pose_w.torch[:, 2] - scene.env_origins[:, 2]
        assert math.isfinite(float(height.min())) and float(height.min()) > 0.3
    finally:
        sim_utils.SimulationContext.clear_instance()
