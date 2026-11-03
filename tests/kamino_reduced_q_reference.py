"""Frozen old q primitive, used only with explicitly bounded dense cross-factor input."""

import warp as wp
from newton._src.solvers.kamino._src.linalg.factorize.llt_blocked import get_float32_array_offset_ptr


def make_reduced_q(block=32):
    @wp.kernel(module="unique", enable_backward=False)
    def solve(
        L: wp.array[wp.float32],
        dims: wp.array[wp.int32],
        mio: wp.array[wp.int32],
        v: wp.array[wp.float32],
        q: wp.array[wp.float32],
        free: int,
        max_problem: int,
        max_reduced: int,
        result: wp.array[wp.float32],
    ):
        w, lane = wp.tid()
        n = dims[w]
        r = n - free
        lm = wp.array(ptr=get_float32_array_offset_ptr(L, mio[w]), shape=(n, n), dtype=wp.float32)
        vv = wp.array(ptr=get_float32_array_offset_ptr(v, w * free), shape=(free, 1), dtype=wp.float32)
        qq = wp.array(ptr=get_float32_array_offset_ptr(q, w * max_problem), shape=(n, 1), dtype=wp.float32)
        out = wp.array(
            ptr=get_float32_array_offset_ptr(result, w * max_reduced), shape=(r, 1), dtype=wp.float32
        )
        for i in range(0, (r + block - 1) // block * block, block):
            rhs = wp.tile_load(qq, shape=(block, 1), offset=(free + i, 0), storage="shared")
            for j in range(0, (free + block - 1) // block * block, block):
                lb = wp.tile_load(lm, shape=(block, block), offset=(free + i, j))
                vb = wp.tile_load(vv, shape=(block, 1), offset=(j, 0))
                wp.tile_matmul(lb, vb, rhs, alpha=-1.0)
            wp.tile_store(out, rhs, offset=(i, 0))

    return solve
