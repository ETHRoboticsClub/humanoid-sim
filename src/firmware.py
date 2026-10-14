"""Batched behavioral JVPT drive slice, independent of Isaac and bus transport.

Every firmware boundary receives due frames in submission order, evaluates the
watchdog, then evaluates effort. One state transition occurs per received frame.
Watchdog expiry latches a modeled fault and immediately removes motor effort;
this chosen policy is not a claim about unmeasured HEJ watchdog behavior.
"""

from dataclasses import dataclass

import torch

from .actuators import ActuatorModel
from .clock import validate_tick_ns
from .contracts import CommandBatch, DriveFeedback, DriveState, MODE_JVPT, SUPPORTED_CONTROLWORDS
from .sensors import SensorBank


@dataclass
class _Pending:
    command: CommandBatch
    due_ns: torch.Tensor
    mask: torch.Tensor
    latest_due_ns: int


class DeviceBank:
    """Drive state is batched [environment, fixed device]. No environment loops.

    ``advance`` must visit every firmware tick with fresh physical feedback.
    Intermediate physics calls return held effort. The first tick is zero.
    Full reset may establish a new epoch; selective reset preserves global time.
    ``watchdog_ns=0`` explicitly disables watchdog modeling.
    ``check_finite=False`` avoids GPU synchronization when a backend supplies
    internal physical tensors. Shape/device/dtype checks remain; commands
    always retain finite validation at the input boundary.
    """

    def __init__(
        self,
        configs,
        num_envs,
        device="cpu",
        dtype=torch.float32,
        firmware_period_ns=400_000,
        queue_capacity=32,
        check_finite=True,
    ):
        self.configs = tuple(configs)
        if not self.configs or len({c.name for c in self.configs}) != len(self.configs):
            raise ValueError("configs must contain distinct named drives")
        if type(num_envs) is not int or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        validate_tick_ns(firmware_period_ns, "firmware_period_ns")
        if not firmware_period_ns:
            raise ValueError("firmware_period_ns must be positive")
        if type(queue_capacity) is not int or queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")
        self.num_envs, self.num_devices = num_envs, len(self.configs)
        self.shape = (num_envs, self.num_devices)
        self.device, self.dtype = torch.device(device), dtype
        self.firmware_period_ns, self.queue_capacity = firmware_period_ns, queue_capacity
        if type(check_finite) is not bool:
            raise ValueError("check_finite must be bool")
        self.check_finite = check_finite
        self.sensors = SensorBank(self.shape, self.device, dtype)
        # Resolve a generic "cuda" device to the concrete allocation index.
        self.device = self.sensors.position.device
        self.sensors.device = self.device
        self.actuators = ActuatorModel(self.configs, self.device, dtype, firmware_period_ns)
        self._delay = torch.tensor(
            [c.command_delay_ns for c in self.configs], device=self.device, dtype=torch.int64
        )[None, :]
        self._max_delay_ns = max(c.command_delay_ns for c in self.configs)
        self._watchdog = torch.tensor(
            [c.watchdog_ns for c in self.configs], device=self.device, dtype=torch.int64
        )[None, :]
        for key in (
            "commanded_position",
            "commanded_velocity",
            "commanded_effort",
            "demand_position",
            "demand_velocity",
            "demand_effort",
            "requested_effort",
            "actuator_effort",
            "friction_effort",
            "applied_effort",
        ):
            setattr(self, key, torch.zeros(self.shape, device=self.device, dtype=dtype))
        self.statusword = torch.full(
            self.shape, int(DriveState.SWITCH_ON_DISABLED), device=self.device, dtype=torch.int64
        )
        self.mode_display = torch.zeros_like(self.statusword)
        self.watchdog_expired = torch.zeros(self.shape, device=self.device, dtype=torch.bool)
        self._last_rx = torch.full_like(self.statusword, -1)
        self._pending = []
        self._now_ns, self._next_tick_ns, self._last_submission_ns = 0, 0, -1
        self._feedback_timestamp_ns, self._firmware_timestamp_ns = 0, -1

    def _validate_command(self, command):
        if not isinstance(command, CommandBatch):
            raise TypeError("command must be CommandBatch")
        for key in ("position", "velocity", "effort"):
            self.sensors.validate(getattr(command, key), key)
        for key in ("controlword", "mode"):
            value = getattr(command, key)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != self.shape:
                raise ValueError(f"{key} must be a Tensor of shape {self.shape}")
            dtypes = (
                (torch.int8, torch.int16, torch.int32, torch.int64)
                if key == "mode"
                else (torch.uint8, torch.int16, torch.int32, torch.int64)
            )
            if value.device != self.device or value.dtype not in dtypes:
                raise ValueError(f"{key} must be an integer Tensor on {self.device}")
        if not bool((command.mode == MODE_JVPT).all()):
            raise NotImplementedError("only cyclic JVPT mode -64 is modeled")
        supported = torch.zeros_like(command.controlword, dtype=torch.bool)
        for value in SUPPORTED_CONTROLWORDS:
            supported |= command.controlword == value
        if not bool(supported.all()):
            raise NotImplementedError("unsupported controlword in frame")

    @torch.no_grad()
    def submit(self, command: CommandBatch, tick_ns: int) -> None:
        """Queue a frame arriving at tick_ns; fixed per-device delay follows.

        Arrivals must be submitted in nondecreasing time order. A full queue
        raises before any state changes. Delivered frames are not wire-decoded.
        """
        validate_tick_ns(tick_ns)
        self._validate_command(command)
        if tick_ns < self._now_ns or tick_ns < self._last_submission_ns:
            raise ValueError("frame arrival time cannot move backward")
        if tick_ns > torch.iinfo(torch.int64).max - self._max_delay_ns:
            raise ValueError("frame time plus delay exceeds int64")
        if len(self._pending) >= self.queue_capacity:
            raise OverflowError("pending command queue is full")
        copied = CommandBatch(
            *(
                getattr(command, key).detach().clone()
                for key in ("controlword", "mode", "position", "velocity", "effort")
            )
        )
        self._pending.append(
            _Pending(
                copied,
                tick_ns + self._delay,
                torch.ones(self.shape, device=self.device, dtype=torch.bool),
                tick_ns + self._max_delay_ns,
            )
        )
        for key in ("position", "velocity", "effort"):
            getattr(self, "commanded_" + key).copy_(getattr(copied, key))
        self._last_submission_ns = tick_ns

    def _deliver(self, frame, mask, now_ns):
        cw = frame.controlword
        previous = self.statusword
        fault = previous == int(DriveState.FAULT)
        state = previous.clone()
        # Fault remains latched until a distinct reset command.
        state = torch.where(mask & fault & (cw == 128), int(DriveState.SWITCH_ON_DISABLED), state)
        active = mask & ~fault
        state = torch.where(active & (cw == 0), int(DriveState.SWITCH_ON_DISABLED), state)
        state = torch.where(active & (cw == 6), int(DriveState.READY_TO_SWITCH_ON), state)
        switch = (
            active
            & (cw == 7)
            & (
                (previous == int(DriveState.READY_TO_SWITCH_ON))
                | (previous == int(DriveState.OPERATION_ENABLED))
            )
        )
        state = torch.where(switch, int(DriveState.SWITCHED_ON), state)
        enable = active & (cw == 15) & (previous == int(DriveState.SWITCHED_ON))
        state = torch.where(enable, int(DriveState.OPERATION_ENABLED), state)
        quick = (
            active
            & (cw == 2)
            & ((previous == int(DriveState.OPERATION_ENABLED)) | (previous == int(DriveState.SWITCHED_ON)))
        )
        state = torch.where(quick, int(DriveState.QUICK_STOP_ACTIVE), state)
        self.statusword.copy_(state)
        for key in ("position", "velocity", "effort"):
            target = getattr(self, "demand_" + key)
            target.copy_(torch.where(mask, getattr(frame, key), target))
        self.mode_display.copy_(torch.where(mask, frame.mode, self.mode_display))
        self._last_rx.copy_(torch.where(mask, now_ns, self._last_rx))
        self.watchdog_expired &= ~(mask & fault & (cw == 128))

    @torch.no_grad()
    def advance(self, now_ns: int, position, velocity):
        validate_tick_ns(now_ns, "now_ns")
        self.sensors.validate(position, "position", check_finite=self.check_finite)
        self.sensors.validate(velocity, "velocity", check_finite=self.check_finite)
        if now_ns < self._now_ns:
            raise ValueError("time cannot move backward")
        if now_ns > self._next_tick_ns:
            raise ValueError(f"skipped firmware tick {self._next_tick_ns}; integrate each boundary")
        if now_ns < self._feedback_timestamp_ns:
            raise ValueError("physical sample time cannot move backward")
        self.sensors.update(position, velocity, check_finite=False)
        self._feedback_timestamp_ns = now_ns
        self._now_ns = now_ns
        if now_ns != self._next_tick_ns:
            return self.applied_effort.clone()
        for pending in self._pending:
            due = pending.mask & (pending.due_ns <= now_ns)
            self._deliver(pending.command, due, now_ns)
            pending.mask &= ~due
        self._pending = [p for p in self._pending if p.latest_due_ns > now_ns]
        expired = (self._last_rx >= 0) & (self._watchdog > 0) & ((now_ns - self._last_rx) >= self._watchdog)
        self.watchdog_expired |= expired
        self.statusword.copy_(torch.where(expired, int(DriveState.FAULT), self.statusword))
        enabled = self.statusword == int(DriveState.OPERATION_ENABLED)
        values = self.actuators.evaluate(
            position,
            velocity,
            self.demand_position,
            self.demand_velocity,
            self.demand_effort,
            enabled,
            self.actuator_effort,
        )
        for name, value in zip(
            ("requested_effort", "actuator_effort", "friction_effort", "applied_effort"), values
        ):
            getattr(self, name).copy_(value)
        self._next_tick_ns += self.firmware_period_ns
        self._firmware_timestamp_ns = now_ns
        return self.applied_effort.clone()

    @torch.no_grad()
    def refresh_feedback(self, position, velocity, effort=None, timestamp_ns=None):
        """Update sensors at physical sample time without advancing firmware.

        Pass the post-step physical timestamp explicitly. Omitting it preserves
        the existing sample time for callers replacing the same sample.
        """
        timestamp = self._feedback_timestamp_ns if timestamp_ns is None else timestamp_ns
        validate_tick_ns(timestamp, "timestamp_ns")
        if timestamp < self._feedback_timestamp_ns:
            raise ValueError("physical sample time cannot move backward")
        self.sensors.update(position, velocity, effort, check_finite=self.check_finite)
        self._feedback_timestamp_ns = timestamp

    @property
    def feedback(self):
        names = (
            "commanded_position",
            "commanded_velocity",
            "commanded_effort",
            "demand_position",
            "demand_velocity",
            "demand_effort",
            "requested_effort",
            "actuator_effort",
            "friction_effort",
            "applied_effort",
            "statusword",
            "mode_display",
            "watchdog_expired",
        )
        return DriveFeedback(
            self._feedback_timestamp_ns,
            self.sensors.position.clone(),
            self.sensors.velocity.clone(),
            self.sensors.measured_effort.clone(),
            *(getattr(self, name).clone() for name in names),
            firmware_timestamp_ns=self._firmware_timestamp_ns,
        )

    @torch.no_grad()
    def reset(self, env_ids=None, position=None, velocity=None, now_ns=None):
        """Reset selected environments atomically, including all delayed frames."""
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, device=self.device)
            if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("env_ids must be a one-dimensional integer sequence")
            if bool(((ids < 0) | (ids >= self.num_envs)).any()):
                raise ValueError("env_ids out of range")
        if position is not None:
            self.sensors.validate(position, "position")
        if velocity is not None:
            self.sensors.validate(velocity, "velocity")
        if now_ns is not None:
            validate_tick_ns(now_ns)
            if env_ids is not None or now_ns % self.firmware_period_ns:
                raise ValueError("only full reset can establish a firmware-boundary epoch")
        for name in (
            "commanded_position",
            "commanded_velocity",
            "commanded_effort",
            "demand_position",
            "demand_velocity",
            "demand_effort",
            "requested_effort",
            "actuator_effort",
            "friction_effort",
            "applied_effort",
            "mode_display",
        ):
            getattr(self, name)[ids] = 0
        self.statusword[ids] = int(DriveState.SWITCH_ON_DISABLED)
        self.watchdog_expired[ids] = False
        self._last_rx[ids] = -1
        self.sensors.position[ids] = 0 if position is None else position[ids]
        self.sensors.velocity[ids] = 0 if velocity is None else velocity[ids]
        self.sensors.measured_effort[ids] = float("nan")
        for pending in self._pending:
            pending.mask[ids] = False
        if env_ids is None:
            self._pending.clear()
            epoch = 0 if now_ns is None else now_ns
            self._now_ns, self._next_tick_ns, self._last_submission_ns = epoch, epoch, -1
            self._feedback_timestamp_ns, self._firmware_timestamp_ns = epoch, -1
        else:
            # Reset is an occasional synchronization boundary. Release empty
            # frames now; the normal firmware path uses only host due times.
            self._pending = [p for p in self._pending if bool(p.mask.any())]

    def configure(self, *args, **kwargs):
        raise NotImplementedError("runtime CoE/configuration operations are not modeled")
