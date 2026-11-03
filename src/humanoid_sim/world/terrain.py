"""Terrain for the training scene, as an Isaac Lab terrain-importer config: heightfields and triangle meshes.

This is the mechanism only: which tiles, how many, how hard and any curriculum over them are the task's choice
(humanoid-learning's terrain generators). What is fixed here is what the physics can collide with. Isaac Lab turns a
generated terrain into a heightfield only when every sub-terrain is a heightfield one (``Hf*``) with conversion on,
and keeps it a single triangle mesh otherwise (stairs with vertical risers, ramps with edges, obstacles).
:func:`heightfield_terrain_cfg` accepts only the former and :func:`mesh_terrain_cfg` the latter. Both collide through
Newton's pipeline (``contacts: newton``, the default on MJWarp; always on Kamino): MJWarp's own collides the robot
with heightfields, returning some sole-box contacts metres deep, and not with triangle meshes (docs/world.md). Imports
Isaac Lab lazily.
"""

from __future__ import annotations


def heightfield_terrain_cfg(sub_terrains: dict, *, size_m: float, rows: int, cols: int, border_m: float,
                            curriculum: bool, friction: float = 1.0, horizontal_scale_m: float = 0.1,
                            vertical_scale_m: float = 0.005, seed: int | None = None):
    """A generated terrain of heightfield sub-terrains (``{name: Hf*Cfg}``, proportions as weights) on a grid of
    ``rows`` by ``cols`` square tiles of ``size_m``. With ``curriculum`` rows are difficulty levels, otherwise each
    tile draws its own. With robot friction randomized pass ``friction=0`` (the solver uses the larger friction of a
    contact). The terrain's restitution is 0, the neutral partner of a randomized robot ``restitution`` (the solver
    uses the larger of a contact's two). The grid spacing and height step set the heightfield's resolution in the
    collision pipeline."""
    from isaaclab.terrains.height_field import HfTerrainBaseCfg

    meshes = [n for n, c in sub_terrains.items() if not isinstance(c, HfTerrainBaseCfg) or not c.convert_to_heightfield]
    if meshes:
        raise ValueError(f"sub-terrains {meshes} are not heightfields; use mesh_terrain_cfg for triangle meshes")
    return _terrain_cfg(sub_terrains, size_m, rows, cols, border_m, curriculum, friction, horizontal_scale_m,
                        vertical_scale_m, seed)


def mesh_terrain_cfg(sub_terrains: dict, *, size_m: float, rows: int, cols: int, border_m: float,
                     curriculum: bool, friction: float = 1.0, seed: int | None = None):
    """A generated terrain with triangle-mesh sub-terrains (``{name: Mesh*Cfg}`` of Isaac Lab's
    ``isaaclab.terrains.trimesh``, such as stairs with vertical risers, proportions as weights), as for
    :func:`heightfield_terrain_cfg`. Heightfield sub-terrains may be among them; the whole terrain is then one
    collision mesh, and the heightfield ones are not converted. The robot collides with the triangles themselves, so
    the geometry is exact: a step's edge is where the mesh has it. At least one sub-terrain must be a mesh one: a
    terrain of heightfields only is a :func:`heightfield_terrain_cfg`."""
    from isaaclab.terrains.height_field import HfTerrainBaseCfg

    if all(isinstance(c, HfTerrainBaseCfg) and c.convert_to_heightfield for c in sub_terrains.values()):
        raise ValueError("every sub-terrain is a heightfield; use heightfield_terrain_cfg")
    return _terrain_cfg(sub_terrains, size_m, rows, cols, border_m, curriculum, friction, 0.1, 0.005, seed)


def _terrain_cfg(sub_terrains, size_m, rows, cols, border_m, curriculum, friction, horizontal_scale_m,
                 vertical_scale_m, seed):
    from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
    from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg

    generator = TerrainGeneratorCfg(size=(size_m, size_m), border_width=border_m, num_rows=rows, num_cols=cols,
                                    horizontal_scale=horizontal_scale_m, vertical_scale=vertical_scale_m,
                                    curriculum=curriculum, sub_terrains=sub_terrains, seed=seed)
    return TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="generator", terrain_generator=generator, max_init_terrain_level=0,
        collision_group=-1, visual_material=None,
        physics_material=RigidBodyMaterialBaseCfg(static_friction=friction, dynamic_friction=friction, restitution=0.0))
