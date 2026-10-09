import json
from pathlib import Path

import torch

from coursekit.labs.line_following_course import SENSOR_WIDTH


def wrap(angle):
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi


class Solution:
    """Two-point line geometry, damped steering and short reverse recovery."""

    def __init__(self, artifact_dir: Path, config: dict, seed: int):
        self.settings = json.loads((artifact_dir / "controller.json").read_text())
        self.position = None

    def act(self, observation, reset):
        rgb, time = observation["rgb"], observation["time"]
        encoders, gyro = observation["encoders"], observation["gyro"]
        if self.position is None or len(self.position) != len(time):
            self.position = time.new_zeros((len(time), 2))
            self.anchor = self.position.clone()
            self.last_point = self.position.clone()
            self.ahead = self.position.clone()
            self.retreat = self.position.clone()
            self.previous_encoders = encoders.clone()
            self.previous_time = time.clone()
            self.yaw = torch.zeros_like(time)
            self.line_yaw = torch.zeros_like(time)
            self.speed = torch.zeros_like(time)
            self.target_speed = torch.zeros_like(time)
            self.speed_integral = torch.zeros_like(time)
            self.heading_integral = torch.zeros_like(time)
            self.error = torch.zeros_like(time)
            self.endpoint_distance = torch.zeros_like(time)
            self.caution = torch.zeros_like(time)
            self.direction = torch.ones_like(time)
            self.mode = torch.zeros_like(time, dtype=torch.int64)  # follow, back, align
            self.end_seen = torch.zeros_like(reset)
            self.return_line = torch.zeros_like(reset)
        for state in (
            self.position,
            self.anchor,
            self.last_point,
            self.ahead,
            self.retreat,
            self.yaw,
            self.line_yaw,
            self.speed,
            self.target_speed,
            self.speed_integral,
            self.heading_integral,
            self.error,
            self.endpoint_distance,
            self.caution,
            self.mode,
        ):
            state[reset] = 0
        self.direction[reset] = 1
        self.end_seen[reset] = self.return_line[reset] = False
        self.previous_time[reset] = time[reset]
        self.previous_encoders[reset] = encoders[reset]
        dt = (time - self.previous_time).clamp(min=0, max=0.1)
        # Rolling odometry: encoders are body-relative, so include pitch rate.
        # The wheel radius is public; slip and the planar approximation cause drift.
        ds = 0.035 * ((encoders - self.previous_encoders).mean(dim=1) + gyro[:, 1] * dt)
        dyaw = gyro[:, 2] * dt
        middle = self.yaw + dyaw / 2
        self.position += ds[:, None] * torch.stack((middle.cos(), middle.sin()), dim=1)
        self.yaw += dyaw
        self.speed += dt / (0.04 + dt) * (ds / dt.clamp(min=0.02) - self.speed)

        # Keep one connected stripe, preserving the previously selected branch.
        weight = (0.8 - rgb.mean(dim=-1)).clamp(min=0)
        dark = weight > 0.02
        groups = (dark & ~torch.nn.functional.pad(dark[:, :-1], (1, 0))).cumsum(dim=1)
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
        self.error = torch.where(visible, offset, self.error)
        # Lift each optical intersection (0, e) into the same local ground frame.
        point = self.position + self.error[:, None] * torch.stack(
            (-self.yaw.sin(), self.yaw.cos()), dim=1
        )
        self.anchor[reset] = point[reset]
        self.last_point[reset] = point[reset]
        self.retreat[reset] = point[reset]

        red = (((rgb[..., 0] - rgb[..., 2]) > 0.35) & (weight > 0)).any(dim=1)
        self.end_seen |= red & (self.direction > 0)
        self.endpoint_distance = torch.where(
            self.end_seen & ~visible & (self.direction > 0),
            self.endpoint_distance + ds.clamp(min=0),
            0.0,
        )
        turning_back = (self.endpoint_distance > 0.015) & (self.direction > 0)
        self.direction[turning_back] = -1
        self.anchor[turning_back] = point[turning_back]
        self.return_line |= (self.direction < 0) & visible
        endpoint = self.end_seen & ~self.return_line

        # Two observed line points estimate the tangent without a pixel derivative.
        # Wait for 12 mm separation; renew at 30 mm. Modulo pi prevents a noisy
        # measurement from accidentally reversing the inferred route orientation.
        vector = (point - self.anchor) * self.direction[:, None]
        span = torch.linalg.vector_norm(vector, dim=1)
        reliable = visible & (self.error.abs() < 0.037)
        fit = reliable & (span > 0.012) & (self.mode == 0) & ~endpoint
        measured = torch.atan2(vector[:, 1], vector[:, 0])
        delta = (measured - self.line_yaw + torch.pi / 2) % torch.pi - torch.pi / 2
        self.line_yaw += torch.where(fit, 0.5 * delta, 0.0)
        renew = reliable & (span > 0.03) & (self.mode == 0)
        self.retreat[renew] = self.anchor[renew]
        self.anchor[renew] = point[renew]

        # Retreat before fitting a new direction. Keep an older observed point,
        # since simply reacquiring paint near the last observation can skip a gap.
        lost = (~visible | (self.error.abs() > 0.040)) & ~endpoint & (self.mode != 1)
        self.ahead[lost & (self.mode == 0)] = self.last_point[lost & (self.mode == 0)]
        self.anchor[lost & (self.mode == 0)] = self.retreat[lost & (self.mode == 0)]
        self.mode[lost] = 1
        backing = self.mode == 1
        back_vector = (self.anchor - self.position) * -self.direction[:, None]
        back_angle = wrap(torch.atan2(back_vector[:, 1], back_vector[:, 0]) - self.yaw)
        found = backing & reliable & (torch.linalg.vector_norm(back_vector, dim=1) < 0.012)
        # The last point before loss is now ahead: turn towards that observed continuation.
        vector = (self.ahead - point) * self.direction[:, None]
        measured = torch.atan2(vector[:, 1], vector[:, 0])
        delta = (measured - self.line_yaw + torch.pi / 2) % torch.pi - torch.pi / 2
        self.line_yaw += torch.where(found, delta, 0.0)
        self.mode[found] = 2
        self.anchor[found] = point[found]
        self.heading = wrap(self.line_yaw - self.yaw)
        aligned = (self.mode == 2) & visible & (self.heading.abs() < 0.15)
        self.mode[aligned] = 0
        self.anchor[aligned] = point[aligned]
        self.caution = torch.where(self.mode != 0, 1.5, (self.caution - dt).clamp(min=0))
        self.last_point[reliable & (self.mode == 0)] = point[reliable & (self.mode == 0)]

        # PID heading feedback uses the gyro for damping, without differentiating pixels.
        heading_error = (
            self.heading + self.direction * self.settings["offset_gain"] * self.error
        )
        self.heading_integral = (self.heading_integral + dt * heading_error).clamp(
            -0.4, 0.4
        )
        self.heading_integral[self.mode != 0] = 0
        target_yaw = (
            self.settings["heading_gain"] * heading_error + 0.2 * self.heading_integral
        ).clamp(-2.0, 2.0)
        target_yaw = torch.where(
            self.mode == 1, (3 * back_angle).clamp(-1.2, 1.2), target_yaw
        )
        target_yaw = torch.where(
            self.mode == 2, (4 * self.heading).clamp(-1.0, 1.0), target_yaw
        )
        target_yaw = torch.where(~visible & endpoint, 0.0, target_yaw)
        target_speed = (
            self.direction
            * self.settings["speed"]
            / (1 + (self.error / 0.02).square() + self.heading.square() * 3)
        )
        target_speed /= (weight.sum(dim=1) / 1.5).clamp(min=1).square()
        target_speed = torch.where(
            self.caution > 0, target_speed.clamp(-0.07, 0.07), target_speed
        )
        target_speed = torch.where(
            self.mode == 1,
            -self.direction * 0.055 * back_angle.cos().clamp(min=0),
            target_speed,
        )
        # v = omega * e pivots approximately about the sensed line intersection.
        target_speed = torch.where(self.mode == 2, gyro[:, 2] * self.error, target_speed)
        target_speed = torch.where(endpoint, self.direction * 0.07, target_speed)
        # Command slew also runs at reset; the odometry still integrates zero time.
        self.target_speed += (target_speed - self.target_speed).clamp(
            -0.7 * dt.clamp(min=0.02), 0.7 * dt.clamp(min=0.02)
        )
        speed_error = self.target_speed - self.speed
        self.speed_integral = (self.speed_integral + dt * speed_error).clamp(-0.08, 0.08)
        throttle = 0.8 * self.target_speed + 0.6 * speed_error + 0.5 * self.speed_integral
        steering = 0.035 * target_yaw + 0.10 * (target_yaw - gyro[:, 2])
        self.command = torch.stack(
            (throttle - steering, throttle + steering), dim=-1
        ).clamp(-0.5, 0.5)
        self.previous_time, self.previous_encoders = time.clone(), encoders.clone()
        self.visible, self.yaw_rate = visible, gyro[:, 2]
        return self.command.clone()

    def debug(self, index=0):
        if self.position is None:
            return {}
        return {
            "line_offset_m": float(self.error[index]),
            "line_visible": bool(self.visible[index]),
            "drive_direction": float(self.direction[index]),
            "heading_error_rad": float(self.heading[index]),
            "recovering": int(self.mode[index]),  # 0: follow, 1: reverse, 2: align
            "speed_m_s": float(self.speed[index]),
            "yaw_rate": float(self.yaw_rate[index]),
            "command": self.command[index].detach().cpu().numpy().copy(),
        }
