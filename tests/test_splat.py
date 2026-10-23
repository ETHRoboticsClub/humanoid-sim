"""The splat converter (``world/splat.py``) against a PLY written here from known Gaussians: the oracle is the 3DGS
file convention (log scales, logit opacities, real-first quaternions, ``f_rest`` channel-major), applied by hand
to the values the test chose. Newton's USD reader, a separate consumer, must read the written file back as the
same Gaussians."""

from __future__ import annotations

import math

import numpy as np
import pytest

from humanoid_sim.world import splat as sp

pxr = pytest.importorskip("pxr")

DEGREE = 3
K = (DEGREE + 1) ** 2
SPAWN = (-4.42, 3.48)


def _gaussians(n: int = 5, seed: int = 3) -> dict:
    """Per-Gaussian values in the units a renderer uses: centres, unit quaternions (w, x, y, z), standard
    deviations, opacities in (0, 1) and RGB spherical harmonic coefficients [n, K, 3]."""
    rng = np.random.default_rng(seed)
    wxyz = rng.normal(size=(n, 4))
    wxyz /= np.linalg.norm(wxyz, axis=1, keepdims=True)
    return {"xyz": rng.uniform(-5, 5, size=(n, 3)), "wxyz": wxyz, "sigma": rng.uniform(0.01, 0.3, size=(n, 3)),
            "alpha": rng.uniform(0.05, 0.95, size=n), "sh": rng.normal(size=(n, K, 3)),
            "quat_scale": rng.uniform(0.5, 3.0, size=n)}


def _write_ply(path, g: dict, *, fmt: str = "binary_little_endian", extra_header: str = "") -> None:
    """A 3DGS PLY of the Gaussians ``g``: ``scale = log(sigma)``, ``opacity = logit(alpha)``, an unnormalized
    quaternion (``quat_scale`` times the unit one), ``f_rest`` the red channel's coefficients 1..K-1, then green's,
    then blue's."""
    n = len(g["xyz"])
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(3 * (K - 1))]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    rows = np.zeros((n, len(names)), dtype=np.float32)
    col = {name: i for i, name in enumerate(names)}
    rows[:, [col["x"], col["y"], col["z"]]] = g["xyz"]
    for c in range(3):
        rows[:, col[f"f_dc_{c}"]] = g["sh"][:, 0, c]
        for k in range(1, K):
            rows[:, col[f"f_rest_{c * (K - 1) + k - 1}"]] = g["sh"][:, k, c]
    rows[:, col["opacity"]] = np.log(g["alpha"] / (1 - g["alpha"]))
    for a in range(3):
        rows[:, col[f"scale_{a}"]] = np.log(g["sigma"][:, a])
    for i in range(4):
        rows[:, col[f"rot_{i}"]] = g["wxyz"][:, i] * g["quat_scale"]
    header = f"ply\nformat {fmt} 1.0\n{extra_header}element vertex {n}\n"
    header += "".join(f"property float {name}\n" for name in names) + "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(rows.tobytes())


def _read_usd(path):
    from pxr import Usd

    stage = Usd.Stage.Open(str(path))
    prim = stage.GetPrimAtPath(sp.PRIM_PATH)
    get = lambda name: prim.GetAttribute(name).Get()  # noqa: E731
    return stage, prim, get


def test_the_ply_convention_read_by_hand(tmp_path):
    g = _gaussians()
    _write_ply(tmp_path / "s.ply", g)
    s = sp.read_ply(tmp_path / "s.ply")
    assert len(s) == 5 and s.sh_degree == DEGREE
    np.testing.assert_allclose(s.positions, g["xyz"], atol=1e-5)
    np.testing.assert_allclose(s.scales, g["sigma"], rtol=1e-5)
    np.testing.assert_allclose(s.opacities, g["alpha"], atol=1e-6)
    np.testing.assert_allclose(s.sh, g["sh"], atol=1e-6)
    # The same rotation: q and -q are one; xyzw is (x, y, z, w) of the unit quaternion.
    expect = g["wxyz"][:, [1, 2, 3, 0]]
    sign = np.sign(np.sum(s.orientations_xyzw * expect, axis=1, keepdims=True))
    np.testing.assert_allclose(s.orientations_xyzw * sign, expect, atol=1e-6)


@pytest.mark.parametrize("half", [False, True])
def test_the_usd_holds_the_placed_gaussians(tmp_path, half):
    g = _gaussians()
    _write_ply(tmp_path / "s.ply", g)
    info = sp.convert(tmp_path / "s.ply", tmp_path / "s.usda", spawn_m=SPAWN, half=half)
    assert info["gaussians"] == 5 and info["sh_degree"] == DEGREE and info["spawn_m"] == list(SPAWN)
    stage, prim, get = _read_usd(tmp_path / "s.usda")
    assert prim.GetTypeName() == "ParticleField3DGaussianSplat"
    assert stage.GetDefaultPrim().GetPath().pathString == "/Splat"
    pos = np.array(get("positions"))
    np.testing.assert_allclose(pos, g["xyz"] - np.array([SPAWN[0], SPAWN[1], 0.0]), atol=1e-5)
    np.testing.assert_allclose(np.array(get("scales")), g["sigma"], rtol=1e-5)
    np.testing.assert_allclose(np.array(get("opacities")), g["alpha"], atol=1e-6)
    q = np.array(get("orientations"))  # numpy layout of a QuatfArray: (x, y, z, w)
    expect = g["wxyz"][:, [1, 2, 3, 0]]
    np.testing.assert_allclose(q * np.sign(np.sum(q * expect, axis=1, keepdims=True)), expect, atol=1e-6)
    assert get("radiance:sphericalHarmonicsDegree") == DEGREE
    name = "radiance:sphericalHarmonicsCoefficients" + ("h" if half else "")
    other = "radiance:sphericalHarmonicsCoefficients" + ("" if half else "h")
    assert not prim.GetAttribute(other).HasValue()
    sh = np.array(get(name), dtype=np.float64).reshape(5, K, 3)  # per particle, degree 0 first
    np.testing.assert_allclose(sh, g["sh"], atol=2e-3 if half else 1e-6, rtol=1e-3 if half else 0)
    lo, hi = (np.array(v) for v in get("extent"))
    pad = sp.EXTENT_SIGMAS * g["sigma"].max()
    np.testing.assert_allclose(lo, pos.min(axis=0) - pad, atol=1e-5)
    np.testing.assert_allclose(hi, pos.max(axis=0) + pad, atol=1e-5)
    binding = pxr.UsdShade.MaterialBindingAPI(prim).GetDirectBinding().GetMaterialPath()
    assert binding.pathString == sp.MATERIAL_PATH


def test_a_lower_degree_keeps_the_first_coefficients(tmp_path):
    g = _gaussians()
    _write_ply(tmp_path / "s.ply", g)
    sp.convert(tmp_path / "s.ply", tmp_path / "s.usda", sh_degree=1)
    stage, _, get = _read_usd(tmp_path / "s.usda")
    assert get("radiance:sphericalHarmonicsDegree") == 1
    sh = np.array(get("radiance:sphericalHarmonicsCoefficients")).reshape(5, 4, 3)
    np.testing.assert_allclose(sh, g["sh"][:, :4], atol=1e-6)
    with pytest.raises(ValueError, match="degrees 0..3"):
        sp.placed(sp.read_ply(tmp_path / "s.ply"), sh_degree=4)


def test_newton_reads_the_file_as_the_same_gaussians(tmp_path):
    newton = pytest.importorskip("newton")
    from pxr import Usd

    g = _gaussians()
    _write_ply(tmp_path / "s.ply", g)
    sp.convert(tmp_path / "s.ply", tmp_path / "s.usdc", spawn_m=SPAWN)
    stage = Usd.Stage.Open(str(tmp_path / "s.usdc"))
    prim = stage.GetPrimAtPath(sp.PRIM_PATH)
    gauss = newton.Gaussian.create_from_usd(prim)
    assert gauss.count == 5 and gauss.sh_degree == DEGREE
    np.testing.assert_allclose(gauss.positions, g["xyz"] - np.array([SPAWN[0], SPAWN[1], 0.0]), atol=1e-5)
    expect = g["wxyz"][:, [1, 2, 3, 0]]  # Newton's rotations are (x, y, z, w)
    rot = np.asarray(gauss.rotations)
    np.testing.assert_allclose(rot * np.sign(np.sum(rot * expect, axis=1, keepdims=True)), expect, atol=1e-6)
    np.testing.assert_allclose(gauss.scales, g["sigma"], rtol=1e-5)
    np.testing.assert_allclose(gauss.opacities, g["alpha"], atol=1e-6)
    # Newton's renderer takes a Gaussian's colour from its first three coefficients: the degree-0 RGB.
    np.testing.assert_allclose(np.asarray(gauss.sh_coeffs)[:, :3], g["sh"][:, 0], atol=1e-6)


def test_files_that_are_no_splat_are_refused(tmp_path):
    g = _gaussians(n=2)
    _write_ply(tmp_path / "ascii.ply", g, fmt="ascii")
    with pytest.raises(ValueError, match="binary_little_endian"):
        sp.read_ply(tmp_path / "ascii.ply")
    (tmp_path / "lfs.ply").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:0\nsize 1\n")
    with pytest.raises(ValueError, match="git-lfs pointer"):
        sp.read_ply(tmp_path / "lfs.ply")
    (tmp_path / "mesh.ply").write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\n"
                                        b"property float y\nproperty float z\nend_header\n" + bytes(12))
    with pytest.raises(ValueError, match="not a 3DGS splat"):
        sp.read_ply(tmp_path / "mesh.ply")
    (tmp_path / "face.ply").write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty float x\n"
                                        b"element face 0\nproperty list uchar int vertex_indices\nend_header\n")
    with pytest.raises(ValueError, match="only vertices"):
        sp.read_ply(tmp_path / "face.ply")


def test_the_degree_follows_the_rest_count(tmp_path):
    g = _gaussians(n=2)
    _write_ply(tmp_path / "s.ply", g)
    assert sp.read_ply(tmp_path / "s.ply").sh_degree == int(math.isqrt(3 * (K - 1) // 3 + 1) - 1) == DEGREE


def test_parts_are_written_under_xforms_that_move_them(tmp_path):
    from pxr import Gf, Usd, UsdGeom

    g = _gaussians(n=6)
    _write_ply(tmp_path / "s.ply", g)
    room = sp.read_ply(tmp_path / "s.ply")
    first = g["xyz"][0]
    rest, parts = sp.split(room, {"hook": [{"center": first.tolist(), "half": [1e-3] * 3}]})
    sp.write_usd(rest, tmp_path / "s.usda", parts=parts)
    stage = Usd.Stage.Open(str(tmp_path / "s.usda"))
    assert len(stage.GetPrimAtPath(sp.PRIM_PATH).GetAttribute("positions").Get()) == 5
    hook = stage.GetPrimAtPath("/Splat/hook/Gaussians")
    np.testing.assert_allclose(np.array(hook.GetAttribute("positions").Get()), [first], atol=1e-5)
    op = UsdGeom.Xformable(stage.GetPrimAtPath("/Splat/hook")).GetOrderedXformOps()[0]
    op.Set(Gf.Vec3d(0.0, 0.0, -1.0))
    world = UsdGeom.Xformable(hook).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    assert tuple(world.ExtractTranslation()) == (0.0, 0.0, -1.0)
