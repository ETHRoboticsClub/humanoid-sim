"""Lazy GPU regression entrypoint; core-only collection requires no Warp/Newton."""

import pytest

pytestmark = pytest.mark.physics


@pytest.mark.parametrize("free", [17, 161, 180])
def test_partial_free_q_poison_and_changing_dimension_graph(free):
    from kamino_reduced_q_fixture import run_partial_free_q_poison_and_changing_dimension_graph

    run_partial_free_q_poison_and_changing_dimension_graph(free)
