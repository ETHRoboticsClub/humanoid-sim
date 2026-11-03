"""Opt-in ownership lifecycle through real Newton classes on CPU and CUDA."""

import pytest

pytestmark = pytest.mark.physics


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize(
    "case_name",
    [
        "test_config_default_and_scope",
        "test_default_has_no_owner_and_actual_preconditioning_exact",
        "test_all_public_routes_preserve_objects_and_publish",
        "test_eager_default_public_getters_remain_allowed_during_capture",
        "test_all_public_first_getters_inside_false_capture_reject",
        "test_data_rebinding_uses_eager_arrays_without_repair",
        "test_operator_rebinding_preserves_detached_alias",
        "test_public_regularize_and_zero_materialize_and_reset",
        "test_actual_repeated_precondition",
        "test_actual_build_reset_and_additive_preserve_source_operator",
        "test_native_sigma_status_and_repeated_proximal_updates",
        "test_full_solver_on_unpublished_lazy_owner_matches_eager",
        "test_refinalize_replaces_owner_and_preserves_old_public_array",
    ],
)
def test_lazy_preconditioning_preserves_operator_lifecycle(device, case_name):
    import warp as wp
    from kamino_lazy_fixture import Tests

    wp.init()
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA is required for the graph/stream ownership lane")
    case = Tests(methodName=case_name)
    case.device = device
    with wp.ScopedDevice(device):
        getattr(case, case_name)()
