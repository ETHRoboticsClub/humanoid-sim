"""Terrains for the training scene, as Isaac Lab terrain-importer configs.

The robot's contacts run in MJWarp's own collision pipeline, which collides the robot's hulls and boxes with
heightfields but not with triangle-mesh terrain. Isaac Lab turns a generated terrain into a heightfield only
when every sub-terrain is a heightfield one (``Hf*``), so :func:`heightfield_terrain_cfg` accepts only those.
Imports Isaac Lab lazily.
"""

from __future__ import annotations


def heightfield_terrain_cfg(sub_terrains: dict, *, size_m: float = 8.0, rows: int = 10, cols: int = 20,
                            border_m: float = 20.0, curriculum: bool = True, friction: float = 1.0,
                            horizontal_scale_m: float = 0.1, vertical_scale_m: float = 0.005, seed: int | None = None):
    """A generated terrain of heightfield sub-terrains (``{name: Hf*Cfg}``, proportions as weights). Rows are
    difficulty levels when ``curriculum``. With robot friction randomized pass ``friction=0`` (MJWarp uses the
    larger friction of a contact)."""
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


def rough_terrain_cfg(*, max_noise_m: float = 0.025, max_obstacle_m: float = 0.0125, flat: float = 0.5, **kwargs):
    """Flat, random-uniform and box-obstacle tiles (proportions ``flat`` and the rest split evenly), roughness
    growing with the row (difficulty) up to ``max_noise_m`` and ``max_obstacle_m``."""
    from isaaclab.terrains.height_field import (
        HfDiscreteObstaclesTerrainCfg,
        HfPyramidSlopedTerrainCfg,
        HfRandomUniformTerrainCfg,
    )

    rest = (1.0 - flat) / 2
    step = 0.005
    return heightfield_terrain_cfg({
        "flat": HfPyramidSlopedTerrainCfg(proportion=flat, slope_range=(0.0, 0.0), platform_width=2.0),
        "uniform": HfRandomUniformTerrainCfg(proportion=rest, noise_range=(0.0, max_noise_m), noise_step=step),
        "obstacles": HfDiscreteObstaclesTerrainCfg(proportion=rest, obstacle_width_range=(0.25, 0.75),
                                                   obstacle_height_range=(0.0, max_obstacle_m), num_obstacles=40),
    }, **kwargs)
