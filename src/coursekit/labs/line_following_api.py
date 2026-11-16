"""Robot observation and command validation without loading the simulator."""

import torch

from coursekit.common import CourseError

OBSERVATION_SHAPES = {
    "time": (),
    "rgb": (16, 3),
    "gyro": (3,),
    "accel": (3,),
    "encoders": (2,),
}


def validate_observation(observation: dict, reset: torch.Tensor) -> None:
    if not isinstance(reset, torch.Tensor) or reset.dtype != torch.bool or reset.ndim != 1:
        raise CourseError("reset must be a Boolean tensor of shape (B,).")
    if not isinstance(observation, dict) or set(observation) != set(OBSERVATION_SHAPES):
        raise CourseError("Expected time, rgb, gyro, accel, and encoders observations.")
    for name, shape in OBSERVATION_SHAPES.items():
        value = observation[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or value.shape != (len(reset), *shape)
            or not torch.isfinite(value).all()
        ):
            raise CourseError(f"Invalid {name} observation shape, dtype, or values.")
    if ((observation["rgb"] < 0) | (observation["rgb"] > 1)).any():
        raise CourseError("RGB observations must be in [0, 1].")


def validate_action(action: torch.Tensor, size: int) -> None:
    if (
        not isinstance(action, torch.Tensor)
        or action.dtype != torch.float32
        or action.shape != (size, 2)
        or not torch.isfinite(action).all()
        or (action.abs() > 1).any()
    ):
        raise CourseError("act() must return finite float32 commands (B, 2) in [-1, 1].")
