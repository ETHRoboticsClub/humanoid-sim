"""Where the gantry ropes attach: the robot description's ``gantry_attachment_link`` frames, else the profile's harness.

The robot description may carry the attachment points of the lab gantry as links (USD prims or URDF links) named
``gantry_attachment_link``, or several as ``gantry_attachment_link_<n>``. :func:`resolve_harness` is the one place that
reads them; training (:class:`~humanoid_sim.world.supports.Gantry` as the plant builds it), the runtime (``serve
--support``) and replay (through ``SimCore``) all take their ropes from the :class:`Harness` it returns.

With links, one rope hangs on each, on the body that carries them (the nearest body the physics keeps, as for sensor
frames), at the link's pose there; the load is split equally over them, as it is over the shoulders. Without links the
profile's ``support.harness`` (``{body, shoulders: [l, r]}``) applies unchanged. No names are typed in this repo for the
links beyond the pattern.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

LINK_PATTERN = re.compile(r"gantry_attachment_link(?:_(\d+))?")


def attachment_link_names(names: Iterable[str]) -> list[str]:
    """The gantry attachment links among ``names`` in rope order.

    A name is ``gantry_attachment_link`` or ``gantry_attachment_link_<n>`` with ``n`` decimal digits. The order is
    numeric, so ``_2`` comes before ``_10``. The bare name, when it comes with numbered ones, is the first rope, ahead
    of every numbered link (even ``_0``). Two names that give the same number (``_1`` and ``_01``) are ambiguous and
    raise ``ValueError``. Other names are ignored.
    """
    found: dict[int, str] = {}
    for name in set(names):
        match = LINK_PATTERN.fullmatch(name)
        if match is None:
            continue
        index = -1 if match.group(1) is None else int(match.group(1))
        if index in found:
            raise ValueError(f"gantry attachment links {sorted([found[index], name])} number the same rope")
        found[index] = name
    return [found[i] for i in sorted(found)]


@dataclass(frozen=True)
class Harness:
    """What a gantry rope hangs on: ``body`` carries the ropes. Either ``shoulders`` (bodies, the profile's
    harness; the rope points are where those bodies are) or ``links`` (the description's attachment links, with
    ``points_b`` their positions in ``body``'s frame, same order); one rope per point."""

    body: str
    shoulders: tuple[str, ...] = ()
    links: tuple[str, ...] = ()
    points_b: tuple[tuple[float, float, float], ...] = ()

    @property
    def num_points(self) -> int:
        return len(self.links) or len(self.shoulders)

    def attachment(self) -> dict:
        """The ``hang_*`` keyword that places the ropes: ``attachment_h`` for links, ``attachment_bodies`` else."""
        if self.links:
            return {"attachment_h": [list(p) for p in self.points_b]}
        return {"attachment_bodies": list(self.shoulders)}

    def describe(self) -> dict:
        """The runtime descriptor's ``harness`` entry: the profile's form without links."""
        if self.links:
            return {"body": self.body, "attachment_links": list(self.links)}
        return {"body": self.body, "shoulders": list(self.shoulders)}


def _description_names(desc) -> set[str]:
    if desc.usd_model is not None:
        return {p.GetName() for p in desc.usd_model.stage.Traverse()}
    return desc.link_names()


def resolve_harness(desc, profile_harness: dict | None) -> Harness | None:
    """The harness of the robot ``desc``: its attachment links when it has any, else ``profile_harness`` (the
    profile's ``support.harness``, ``{body, shoulders}`` or None, for none). All links must ride on one body."""
    links = attachment_link_names(_description_names(desc))
    if not links:
        if profile_harness is None:
            return None
        return Harness(profile_harness["body"], shoulders=tuple(profile_harness["shoulders"]))
    from humanoid_sim.sensors.geometry import Mount

    mounts = [Mount.from_description(desc, link) for link in links]
    bodies = sorted({m.link for m in mounts})
    if len(bodies) != 1:
        raise ValueError(f"gantry attachment links {links} ride on different bodies {bodies}; ropes pull on one body")
    return Harness(bodies[0], links=tuple(links),
                   points_b=tuple(tuple(float(v) for v in m.translation_m) for m in mounts))
