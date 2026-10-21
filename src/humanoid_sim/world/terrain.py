"""Heightfield terrain for the training scene, as an Isaac Lab terrain-importer config.

This is the mechanism only: which tiles, how many, how hard and any curriculum over them are the task's choice
(humanoid-learning's terrain generators). What is fixed here is what the physics can collide with. Isaac Lab turns a
generated terrain into a heightfield only when every sub-terrain is a heightfield one (``Hf*``), and MJWarp's own
collision pipeline (``contacts: mujoco``) collides the robot with heightfields but not with triangle-mesh terrain,
so :func:`heightfield_terrain_cfg` accepts only those. Collide them through Newton's pipeline (``contacts: newton``,
the default): MJWarp's own returns some sole-box contacts metres deep (docs/world.md). Imports Isaac
Lab lazily.
"""

from __future__ import annotations


def heightfield_terrain_cfg(sub_terrains: dict, *, size_m: float, rows: int, cols: int, border_m: float,
                            curriculum: bool, friction: float = 1.0, horizontal_scale_m: float = 0.1,
                            vertical_scale_m: float = 0.005, seed: int | None = None):
    """A generated terrain of heightfield sub-terrains (``{name: Hf*Cfg}``, proportions as weights) on a grid of
    ``rows`` by ``cols`` square tiles of ``size_m``. With ``curriculum`` rows are difficulty levels, otherwise each
    tile draws its own. With robot friction randomized pass ``friction=0`` (the solver uses the larger friction of a
    contact). The grid spacing and height step set the heightfield's resolution in the collision pipeline."""
    from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
    from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
    from isaaclab.terrains.height_field import HfTerrainBaseCfg

    meshes = [n for n, c in sub_terrains.items() if not isinstance(c, HfTerrainBaseCfg) or not c.convert_to_heightfield]
    if meshes:
        raise ValueError(f"sub-terrains {meshes} are not heightfields; MJWarp's contacts need heightfield terrain")
    generator = TerrainGeneratorCfg(size=(size_m, size_m), border_width=border_m, num_rows=rows, num_cols=cols,
                                    horizontal_scale=horizontal_scale_m, vertical_scale=vertical_scale_m,
                                    curriculum=curriculum, sub_terrains=sub_terrains, seed=seed)
    return TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="generator", terrain_generator=generator, max_init_terrain_level=0,
        collision_group=-1, visual_material=None,
        physics_material=RigidBodyMaterialBaseCfg(static_friction=friction, dynamic_friction=friction, restitution=0.0))
