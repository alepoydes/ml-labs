import json
from pathlib import Path

import torch

from coursekit.labs.line_following_course import SENSOR_WIDTH


class Solution:
    """Slow supported driving with a line centroid, PD steering, and reverse return."""

    def __init__(self, artifact_dir: Path, config: dict, seed: int):
        self.settings = json.loads((artifact_dir / "controller.json").read_text())
        self.direction = None

    def act(self, observation, reset):
        rgb, time = observation["rgb"], observation["time"]
        if self.direction is None or len(self.direction) != len(time):
            self.direction = torch.ones_like(time)
            self.previous = torch.zeros_like(time)
            self.previous_time = torch.zeros_like(time)
            self.side = torch.ones_like(time)
            self.end_seen = torch.zeros_like(reset)
            self.return_line = torch.zeros_like(reset)
            self.lost = torch.zeros_like(time, dtype=torch.int64)
        self.direction[reset] = self.side[reset] = 1
        self.previous[reset] = self.previous_time[reset] = 0
        self.end_seen[reset] = False
        self.return_line[reset] = False
        self.lost[reset] = 0

        weight = (0.8 - rgb.mean(dim=-1)).clamp(min=0)
        coordinates = (torch.arange(16, device=rgb.device) + 0.5) / 16 - 0.5
        self.visible = weight.sum(dim=1) > 0.2
        self.error = (weight * coordinates).sum(dim=1) / weight.sum(dim=1).clamp(min=0.01)
        self.error *= SENSOR_WIDTH
        self.side = torch.where(
            self.visible & (self.error.abs() > 0.001), self.error.sign(), self.side
        )
        self.lost = torch.where(self.visible, 0, self.lost + 1)
        self.end_seen |= ((rgb[..., 0] - rgb[..., 2]) > 0.35).any(dim=1)
        self.direction = torch.where(self.end_seen & (self.lost >= 3), -1.0, self.direction)
        self.return_line |= (self.direction < 0) & self.visible

        dt = (time - self.previous_time).clamp(min=0.02)
        derivative = torch.where(reset, 0.0, (self.error - self.previous) / dt)
        self.yaw_rate = observation["gyro"][:, 2]
        steering = self.direction * (
            self.settings["kp"] * self.error + self.settings["kd"] * derivative
        )
        steering -= self.settings["yaw_damping"] * self.yaw_rate
        endpoint_reversal = self.end_seen & ~self.return_line
        searching = ~self.visible & ~endpoint_reversal
        steering = torch.where(
            searching, self.direction * self.side * self.settings["search_turn"], steering
        )
        throttle = self.direction * self.settings["throttle"]
        throttle = torch.where(searching, -0.5 * throttle, throttle)
        self.command = torch.stack(
            (throttle - steering, throttle + steering), dim=-1
        ).clamp(-1, 1)
        self.previous, self.previous_time = self.error.clone(), time.clone()
        return self.command.clone()

    def debug(self, index=0):
        if self.direction is None:
            return {}
        return {
            "line_offset_m": float(self.error[index]),
            "line_visible": bool(self.visible[index]),
            "drive_direction": float(self.direction[index]),
            "yaw_rate": float(self.yaw_rate[index]),
            "command": self.command[index].detach().cpu().numpy().copy(),
        }
