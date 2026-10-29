"""Close the USD ankle rods when initializing a semantic joint pose."""

from __future__ import annotations

import numpy as np
import torch


class NativeState:
    def __init__(self, desc, names, *, device="cpu", dtype=torch.float32):
        from pxr import Usd, UsdGeom, UsdPhysics

        self.names = list(names)
        self.transmissions = {}
        self.rods = []
        stage = Usd.Stage.Open(str(desc.usd_path))
        module = desc.ankle_model_module()
        joints = [module._Joint(prim, UsdGeom.GetStageMetersPerUnit(stage))
                  for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)]

        def tensor(value):
            return torch.as_tensor(np.asarray(value), device=device, dtype=dtype)

        def connected(body):
            return [joint for joint in joints if body in joint.bodies]

        def one(items):
            if len(items) != 1:
                raise ValueError("ambiguous USD rod topology")
            return items[0]

        for linkage in desc.linkages:
            spec = desc.ankle_model.payload["sides"][linkage.side]
            self.transmissions[linkage.side] = (
                [self.names.index(name) for name in linkage.joints],
                [spec["usd_joints"][name] for name in ("upper", "lower")],
                desc.ankle_transmission(linkage.side, device=device, dtype=dtype))
        for closing in (joint for joint in joints if joint.closing and joint.spherical):
            rod = one([body for body in closing.bodies if body and
                       all(joint.spherical for joint in connected(body))])
            foot = closing.other(rod)
            ball = one([joint for joint in connected(rod) if joint is not closing])
            crank = ball.other(rod)
            shaft = one([joint for joint in connected(crank) if joint.revolute])
            shin = shaft.other(crank)
            roll = one([joint for joint in connected(foot) if joint.revolute])
            cross = roll.other(foot)
            pitch = one([joint for joint in connected(cross) if joint is not roll])
            if ball.bodies[0] != crank:
                raise ValueError(f"{ball.name}: rod joint must run from crank to rod")

            def frames(joint, parent):
                index = joint.bodies.index(parent)
                return (tensor(joint.frames[index]), tensor(np.linalg.inv(joint.frames[1 - index])),
                        tensor(joint.axis * (1 if index == 0 else -1)))

            self.rods.append((ball.name, shaft.name, self.names.index(pitch.name), self.names.index(roll.name),
                              frames(pitch, shin), frames(roll, cross), frames(shaft, shin),
                              tensor(ball.frames[0]), tensor(closing.point(foot)),
                              tensor(ball.frames[1][:3, :3].T @ (closing.point(rod) - ball.point(rod)))))

    def coordinates(self, positions):
        result = {}
        for indices, names, transmission in self.transmissions.values():
            angles = transmission.cranks(positions[..., indices])
            result.update({name: angles[..., index] for index, name in enumerate(names)})
        for name, shaft, pitch, roll, pitch_frames, roll_frames, shaft_frames, ball, anchor, rest in self.rods:
            foot_pose = _pose(pitch_frames, positions[..., pitch]) @ _pose(roll_frames, positions[..., roll])
            ball_pose = _pose(shaft_frames, result[shaft]) @ ball
            target = (foot_pose[..., :3, :3] @ anchor.unsqueeze(-1)).squeeze(-1) + foot_pose[..., :3, 3]
            direction = (ball_pose[..., :3, :3].transpose(-1, -2)
                         @ (target - ball_pose[..., :3, 3]).unsqueeze(-1)).squeeze(-1)
            direction = torch.nn.functional.normalize(direction, dim=-1)
            origin = torch.nn.functional.normalize(rest, dim=-1).expand_as(direction)
            vector = torch.linalg.cross(origin, direction)
            scalar = 1 + (origin * direction).sum(-1, keepdim=True)
            result[name] = torch.nn.functional.normalize(torch.cat((vector, scalar), dim=-1), dim=-1)
        return result

    def state(self, positions, velocities):
        values, derivatives = torch.func.jvp(self.coordinates, (positions,), (velocities,))
        result = {}
        for name, value in values.items():
            derivative = derivatives[name]
            if value.ndim == positions.ndim:
                vector, scalar = value[..., :3], value[..., 3:]
                length = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
                factor = torch.where(length > 1e-12, 2 * torch.atan2(length, scalar) / length.clamp_min(1e-12), 2)
                angular = 2 * (scalar * derivative[..., :3] - derivative[..., 3:] * vector
                               + torch.linalg.cross(vector, derivative[..., :3]))
                for axis in range(3):
                    result[f"{name}:{axis}"] = ((factor * vector)[..., axis], angular[..., axis])
            else:
                result[name] = (value, derivative)
        return result


def _pose(frames, angle):
    parent, child, axis = frames
    cross = torch.stack((torch.stack((axis[0] * 0, -axis[2], axis[1])),
                         torch.stack((axis[2], axis[0] * 0, -axis[0])),
                         torch.stack((-axis[1], axis[0], axis[0] * 0))))
    identity = torch.eye(3, device=angle.device, dtype=angle.dtype)
    rotation = (identity + torch.sin(angle)[..., None, None] * cross
                + (1 - torch.cos(angle))[..., None, None] * (cross @ cross))
    transform = torch.eye(4, device=angle.device, dtype=angle.dtype).expand(*angle.shape, 4, 4).clone()
    transform[..., :3, :3] = rotation
    return parent @ transform @ child
