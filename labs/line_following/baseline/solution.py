import json
from pathlib import Path

import torch

from coursekit.labs.line_following_course import SENSOR_WIDTH


class Solution:
    """Supported feedback control with a short encoder history for recovery."""

    def __init__(self, artifact_dir: Path, config: dict, seed: int):
        self.settings = json.loads((artifact_dir / "controller.json").read_text())
        self.direction = None

    def act(self, observation, reset):
        rgb, time = observation["rgb"], observation["time"]
        if self.direction is None or len(self.direction) != len(time):
            self.direction = torch.ones_like(time)
            self.previous_time = torch.zeros_like(time)
            self.previous_encoders = torch.zeros_like(observation["encoders"])
            self.error = torch.zeros_like(time)
            self.heading = torch.zeros_like(time)
            self.speed = torch.zeros_like(time)
            self.integral = torch.zeros_like(time)
            self.end_seen = torch.zeros_like(reset)
            self.return_line = torch.zeros_like(reset)
            self.visible = torch.zeros_like(reset)
            self.target_speed = torch.zeros_like(time)
            self.endpoint_distance = torch.zeros_like(time)
            self.caution = torch.zeros_like(time)
            self.history = observation["encoders"][:, None].repeat(1, 40, 1)
            self.rewind = torch.full_like(time, -1, dtype=torch.int64)
        self.direction[reset] = 1
        self.history[reset] = observation["encoders"][reset, None]
        self.rewind[reset] = -1
        self.previous_time[reset] = time[reset]
        self.previous_encoders[reset] = observation["encoders"][reset]
        for value in (
            self.error,
            self.heading,
            self.speed,
            self.integral,
            self.target_speed,
            self.endpoint_distance,
            self.caution,
        ):
            value[reset] = 0
        self.end_seen[reset] = self.visible[reset] = self.return_line[reset] = False
        dt = (time - self.previous_time).clamp(min=0.02)
        measured_speed = 0.035 * (
            (observation["encoders"] - self.previous_encoders).mean(dim=1) / dt
            + observation["gyro"][:, 1]
        )
        self.speed += dt / (0.06 + dt) * (measured_speed - self.speed)

        # Select one connected dark run, using the last offset to disambiguate branches.
        weight = (0.8 - rgb.mean(dim=-1)).clamp(min=0)
        dark = weight > 0.02
        starts = dark & ~torch.nn.functional.pad(dark[:, :-1], (1, 0))
        groups = starts.cumsum(dim=1)
        coordinates = SENSOR_WIDTH * (
            (torch.arange(16, device=rgb.device) + 0.5) / 16 - 0.5
        )
        distance = (
            (coordinates - self.error[:, None]).abs().masked_fill(~dark, float("inf"))
        )
        selected = groups.gather(1, distance.argmin(dim=1, keepdim=True))
        weight *= (groups == selected) & dark
        visible = weight.sum(dim=1) > 0.2
        offset = (weight * coordinates).sum(dim=1) / weight.sum(dim=1).clamp(min=0.01)
        derivative = torch.where(visible & self.visible, (offset - self.error) / dt, 0.0)
        self.error = torch.where(visible, offset, self.error)

        # Red starts before the endpoint. Leave its paint before a gentle reverse.
        red = (((rgb[..., 0] - rgb[..., 2]) > 0.35) & (weight > 0)).any(dim=1)
        self.end_seen |= red
        self.endpoint_distance = torch.where(
            self.end_seen & ~visible & (self.direction > 0),
            self.endpoint_distance + dt * self.speed.clamp(min=0),
            0.0,
        )
        self.direction = torch.where(self.endpoint_distance > 0.015, -1.0, self.direction)
        self.return_line |= (self.direction < 0) & visible
        reversing = self.end_seen & ~self.return_line

        # For the line intersection (0, e), e_dot = (v - yaw_rate * e) * tan(heading).
        # Predict with the gyro; correct with optics when the intersection is moving.
        self.yaw_rate = observation["gyro"][:, 2]
        line_speed = self.speed - self.yaw_rate * self.error
        measured_heading = (
            torch.atan(derivative / line_speed.abs().clamp(min=0.025)) * line_speed.sign()
        )
        self.heading -= self.yaw_rate * dt
        heading_delta = (
            measured_heading - self.heading + torch.pi / 2
        ) % torch.pi - torch.pi / 2
        self.heading += torch.where(
            visible & self.visible & (line_speed.abs() > 0.025),
            dt / (0.12 + dt) * heading_delta,
            0.0,
        )
        self.heading = torch.where(visible & ~self.visible, 0.0, self.heading).clamp(
            -1.4, 1.4
        )

        target_yaw = (
            self.direction * self.settings["kp"] * self.error
            + self.settings["heading_gain"] * self.heading
        )
        target_yaw = target_yaw.clamp(-4, 4)
        target_speed = (
            self.direction
            * self.settings["speed"]
            / (1 + 0.6 * self.yaw_rate.abs() + (self.error / 0.015).square())
        )
        target_speed /= (weight.sum(dim=1) / 1.5).clamp(min=1).square()

        lost = ~visible & ~reversing
        self.rewind = torch.where(lost & (self.rewind < 0), 0, self.rewind)
        searching = self.rewind >= 0
        target_yaw = torch.where(~visible & reversing, 0.0, target_yaw)
        target_speed = torch.where(
            ~visible & reversing, self.direction * 0.08, target_speed
        )
        target_speed = torch.where(
            red & (self.direction > 0), target_speed.clamp(max=0.08), target_speed
        )
        self.caution = torch.where(searching, 3.0, (self.caution - dt).clamp(min=0))
        target_speed = torch.where(
            self.caution > 0, target_speed.clamp(-0.045, 0.045), target_speed
        )

        # PI speed control and yaw-rate feedback; ramp speed to retain sphere support.
        self.target_speed += (target_speed - self.target_speed).clamp(-0.7 * dt, 0.7 * dt)
        target_speed = self.target_speed
        speed_error = target_speed - self.speed
        self.integral = (self.integral + dt * speed_error).clamp(-0.08, 0.08)
        throttle = 0.8 * target_speed + 0.6 * speed_error + 0.5 * self.integral
        steering = 0.035 * target_yaw + 0.08 * (target_yaw - self.yaw_rate)
        self.command = torch.stack(
            (throttle - steering, throttle + steering), dim=-1
        ).clamp(-0.5, 0.5)

        # Retrace recent wheel positions, then retry slowly. Slip limits this recovery.
        target = self.history[torch.arange(len(time)), self.rewind.clamp(min=0)]
        wheel_error = target - observation["encoders"]
        wheel_speed = (observation["encoders"] - self.previous_encoders) / dt[:, None]
        undo = (0.3 * wheel_error - 0.025 * wheel_speed).clamp(-0.12, 0.12)
        self.command = torch.where(searching[:, None], undo, self.command)
        reached = wheel_error.abs().amax(dim=1) < 0.10
        self.rewind = torch.where(searching & reached, self.rewind + 1, self.rewind)
        finished = self.rewind == self.history.shape[1]
        self.rewind[finished] = -1
        self.heading[finished] = 0
        self.target_speed[finished] = self.speed[finished]
        self.integral[searching] = 0
        self.history = torch.where(
            searching[:, None, None],
            self.history,
            torch.cat((observation["encoders"][:, None], self.history[:, :-1]), dim=1),
        )
        self.previous_time = time.clone()
        self.previous_encoders = observation["encoders"].clone()
        self.visible = visible
        return self.command.clone()

    def debug(self, index=0):
        if self.direction is None:
            return {}
        return {
            "line_offset_m": float(self.error[index]),
            "line_visible": bool(self.visible[index]),
            "drive_direction": float(self.direction[index]),
            "heading_error_rad": float(self.heading[index]),
            "recovering": bool(self.rewind[index] >= 0),
            "speed_m_s": float(self.speed[index]),
            "yaw_rate": float(self.yaw_rate[index]),
            "command": self.command[index].detach().cpu().numpy().copy(),
        }
