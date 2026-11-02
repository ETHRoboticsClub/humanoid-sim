import pytest

from humanoid_sim.randomization.spec import ParamSpec
from humanoid_sim.train.plant import imu_latency_bounds, imu_latency_ns


def test_without_an_imu_path_samples_reach_the_consumer_as_delivered():
    assert imu_latency_ns({"devices": {}}) == 0


def test_the_nominal_is_the_profiles_measured_path():
    assert imu_latency_ns({"devices": {"imu_path": {"sample_to_host_ns": 2_100_000}}}) == 2_100_000


@pytest.mark.parametrize("total", [-1, 2.1e6, True])
def test_a_path_not_in_whole_nonnegative_ns_is_refused(total):
    with pytest.raises(ValueError, match="sample_to_host_ns"):
        imu_latency_ns({"devices": {"imu_path": {"sample_to_host_ns": total}}})


def test_bounds_follow_the_specs_in_order():
    specs = [ParamSpec("imu_latency", "uniform", (0.5, 2.0), "scale"),
             ParamSpec("imu_latency", "uniform", (0.0, 0.001), "add"),
             ParamSpec("body_mass", "uniform", (0.9, 1.1), "scale")]
    assert imu_latency_bounds(specs, 2_000_000) == pytest.approx((1_000_000, 5_000_000))
    assert imu_latency_bounds([ParamSpec("imu_latency", "uniform", (0.001, 0.003), "set")], 2_000_000) == \
        pytest.approx((1_000_000, 3_000_000))


def test_an_unbounded_draw_is_refused():
    with pytest.raises(ValueError, match="bounded"):
        imu_latency_bounds([ParamSpec("imu_latency", "normal", (1.0, 0.1), "scale")], 2_000_000)
