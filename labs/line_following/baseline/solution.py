import json
from pathlib import Path

import torch

from coursekit.labs.line_following_course import SENSOR_WIDTH


def wrap(angle):
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi


class Solution:
    """Supported driving with two-point line geometry and short reverse recovery."""

    def __init__(self, artifact_dir: Path, config: dict, seed: int):
        self.settings = json.loads((artifact_dir / "controller.json").read_text())
        self.position = None

    def act(self, observation, reset):
        rgb, time = observation["rgb"], observation["time"]
        encoders, gyro = observation["encoders"], observation["gyro"]
        if self.position is None or len(self.position) != len(time):
            self.position = time.new_zeros((len(time), 2))
            self.anchor = self.position.clone()
            self.bookmark = self.position.clone()
            self.retreat = self.position.clone()
            self.previous_encoders = encoders.clone()
            self.previous_time = time.clone()
            self.yaw = torch.zeros_like(time)
            self.line_yaw = torch.zeros_like(time)
            self.bookmark_yaw = torch.zeros_like(time)
            self.retreat_yaw = torch.zeros_like(time)
            self.speed = torch.zeros_like(time)
            self.target_speed = torch.zeros_like(time)
            self.target_yaw = torch.zeros_like(time)
            self.speed_integral = torch.zeros_like(time)
            self.error = torch.zeros_like(time)
            self.endpoint_distance = torch.zeros_like(time)
            self.caution = torch.zeros_like(time)
            self.direction = torch.ones_like(time)
            self.mode = torch.zeros_like(time, dtype=torch.int64)
            self.end_mode = torch.zeros_like(time, dtype=torch.int64)
            self.settled = torch.zeros_like(time)
            self.probe_distance = torch.zeros_like(time)
            self.loss_distance = torch.zeros_like(time)
            self.probe_started = torch.zeros_like(reset)
            self.crossing = torch.zeros_like(reset)
            self.crossing_distance = torch.zeros_like(time)
            self.crossing_clear = torch.zeros_like(time)
            self.crossing_point = self.position.clone()
            self.crossing_yaw = torch.zeros_like(time)
            self.wide = torch.zeros_like(reset)
        for state in (
            self.position,
            self.anchor,
            self.bookmark,
            self.retreat,
            self.yaw,
            self.line_yaw,
            self.bookmark_yaw,
            self.retreat_yaw,
            self.speed,
            self.target_speed,
            self.target_yaw,
            self.speed_integral,
            self.error,
            self.endpoint_distance,
            self.caution,
            self.mode,
            self.end_mode,
            self.settled,
            self.probe_distance,
            self.loss_distance,
            self.crossing_distance,
            self.crossing_clear,
            self.crossing_point,
            self.crossing_yaw,
        ):
            state[reset] = 0
        self.direction[reset] = 1
        self.probe_started[reset] = False
        self.crossing[reset] = False
        self.wide[reset] = False
        self.previous_time[reset] = time[reset]
        self.previous_encoders[reset] = encoders[reset]
        dt = (time - self.previous_time).clamp(min=0, max=0.1)
        # Encoders are body-relative: include pitch rate in rolling odometry.
        # The resting pitch is about -0.25 rad, so gyro Z is yaw rate * cos(0.25).
        # This planar approximation still drifts under slip and changes in tilt.
        ds = 0.035 * ((encoders - self.previous_encoders).mean(dim=1) + gyro[:, 1] * dt)
        self.yaw_rate = gyro[:, 2] / 0.9689124
        dyaw = self.yaw_rate * dt
        middle = self.yaw + dyaw / 2
        self.position += ds[:, None] * torch.stack((middle.cos(), middle.sin()), dim=1)
        self.yaw += dyaw
        self.speed += dt / (0.04 + dt) * (ds / dt.clamp(min=0.02) - self.speed)

        # Normalise darkness by the recognised paint's contrast with the floor.
        # Keep a connected stripe near the previous intersection; averaging across
        # separate branches would place the inferred line in the white gap.
        colour = rgb[..., 0] - rgb[..., 2]
        contrast = torch.where(
            colour > 0.05,
            0.94 - 1.10 / 3,
            torch.where(colour < -0.05, 0.94 - 1.23 / 3, 0.88),
        )
        weight = (0.94 - rgb.mean(dim=-1)).clamp(min=0) * 0.88 / contrast
        dark = weight > 0.025
        groups = (dark & ~torch.nn.functional.pad(dark[:, :-1], (1, 0))).cumsum(dim=1)
        coordinates = SENSOR_WIDTH * (
            (torch.arange(16, device=rgb.device) + 0.5) / 16 - 0.5
        )
        # While crossing merged paint, predict the sensor intersection with the
        # last reliable tangent. The centroid of two branches is not a line point.
        tangent = torch.stack((self.line_yaw.cos(), self.line_yaw.sin()), dim=1)
        normal = torch.stack((-tangent[:, 1], tangent[:, 0]), dim=1)
        heading = wrap(self.line_yaw - self.yaw)
        predicted = -((self.position - self.anchor) * normal).sum(dim=1)
        predicted /= heading.cos().clamp(min=0.35)
        expected = torch.where(self.crossing, predicted, self.error)
        distance = (coordinates - expected[:, None]).abs().masked_fill(~dark, float("inf"))
        selected = groups.gather(1, distance.argmin(dim=1, keepdim=True))
        weight *= (groups == selected) & dark
        visible = weight.sum(dim=1) > 0.2
        offset = (weight * coordinates).sum(dim=1) / weight.sum(dim=1).clamp(min=0.01)
        self.error = torch.where(visible, offset, self.error)
        # Lift (0, offset) into the same local ground frame as the older point.
        point = self.position + self.error[:, None] * torch.stack(
            (-self.yaw.sin(), self.yaw.cos()), dim=1
        )
        self.anchor[reset] = point[reset]

        # Pass the physical end of either coloured stripe before reversing.
        # Initial blue paint is not the outward target. Do not rotate the body.
        colour = rgb[..., 0] - rgb[..., 2]
        target_marker = ((colour * self.direction[:, None] > 0.35) & (weight > 0)).any(
            dim=1
        )
        self.end_mode[(self.end_mode == 0) & target_marker & (self.mode == 0)] = 1
        endpoint = self.end_mode != 0
        self.endpoint_distance = torch.where(
            (self.end_mode == 1) & ~visible,
            self.endpoint_distance + (ds * self.direction).clamp(min=0),
            0.0,
        )
        turning_back = (self.end_mode == 1) & (self.endpoint_distance > 0.012)
        self.direction[turning_back] *= -1
        self.end_mode[turning_back] = 2
        self.anchor[turning_back] = point[turning_back]
        self.end_mode[(self.end_mode == 2) & visible] = 0

        # One stripe covers about two sectors, growing as 1/cos(heading).
        # Save the incoming line BEFORE a wide stripe can corrupt the tangent.
        # Confirm a crossing only when paint extends to both sides of THAT line.
        # A corner or a parallel hairpin extends to one side; track it normally.
        broad = weight.sum(dim=1) > 2.8 / heading.cos().clamp(min=0.35)
        widening = broad & ~self.wide & ~self.crossing
        self.crossing_point[widening] = self.anchor[widening]
        self.crossing_yaw[widening] = self.line_yaw[widening]
        self.wide = broad
        normal = torch.stack((-self.crossing_yaw.sin(), self.crossing_yaw.cos()), dim=1)
        lateral = ((self.position - self.crossing_point) * normal).sum(dim=1)
        cosine = wrap(self.crossing_yaw - self.yaw).cos()
        beyond = lateral[:, None] + coordinates * cosine[:, None]
        crossing_cue = ((weight * (beyond > 0.009)).sum(dim=1) > 0.15) & (
            (weight * (beyond < -0.009)).sum(dim=1) > 0.15
        )
        # Even faint endpoint paint retains ordinary tracking and reversal;
        # do not replace its geometry with a straight crossing probe.
        coloured = ((colour.abs() > 0.05) & (weight > 0.025)).any(dim=1)
        enter_crossing = (
            broad & crossing_cue & (self.mode == 0) & ~endpoint & ~coloured & ~self.crossing
        )
        self.crossing |= enter_crossing
        self.line_yaw[enter_crossing] = self.crossing_yaw[enter_crossing]
        self.anchor[enter_crossing] = self.crossing_point[enter_crossing]
        predicted = torch.where(self.crossing, -lateral / cosine.clamp(min=0.35), predicted)
        self.crossing_distance = torch.where(
            self.crossing & ~enter_crossing, self.crossing_distance + ds.abs(), 0.0
        )
        clear = visible & ~broad & ((offset - predicted).abs() < 0.012)
        self.crossing_clear = torch.where(
            self.crossing & clear, self.crossing_clear + ds.abs(), 0.0
        )
        leave_crossing = self.crossing & (self.crossing_clear > 0.012)
        failed_crossing = (
            self.crossing & (~visible | (self.crossing_distance > 0.14)) & ~leave_crossing
        )
        self.crossing &= ~(leave_crossing | failed_crossing)
        self.anchor[leave_crossing] = point[leave_crossing]

        # Fit over net spatial displacement, not instantaneous encoder speed.
        # Slow pivots can have almost zero v - omega*offset without erasing useful
        # earlier observations. Reject a segment pointing back along the old line.
        vector = (point - self.anchor) * self.direction[:, None]
        span = torch.linalg.vector_norm(vector, dim=1)
        tangent = torch.stack((self.line_yaw.cos(), self.line_yaw.sin()), dim=1)
        advancing = (vector * tangent).sum(dim=1) > -0.0007
        reliable = visible & (self.error.abs() < 0.033) & ~self.crossing & ~failed_crossing
        # Longer separation suppresses optical quantisation near the centre;
        # shorten it when the line approaches an edge to respond to tight bends.
        spacing = torch.where(self.error.abs() < 0.006, 0.016, 0.004)
        beyond_bookmark = (
            (self.position - self.retreat)
            * torch.stack((self.retreat_yaw.cos(), self.retreat_yaw.sin()), dim=1)
        ).sum(dim=1) * self.direction >= 0
        # Crossing the bookmark is an event, not a continuing position test:
        # a hairpin legitimately returns behind the incoming tangent's plane.
        self.probe_started &= self.mode == 3
        self.probe_started |= (self.mode == 3) & beyond_bookmark
        fit = (
            reliable
            & advancing
            & (span > spacing)
            & ((self.mode == 0) | ((self.mode == 3) & self.probe_started))
        )
        measured = torch.atan2(vector[:, 1], vector[:, 0])
        delta = wrap(measured - self.line_yaw)
        # Bound changes in the local tangent: a quantised centroid must not
        # request an instantaneous reversal of the estimated line direction.
        self.line_yaw += torch.where(fit, (0.3 * delta).clamp(-3 * dt, 3 * dt), 0.0)
        self.probe_distance = torch.where(
            self.mode == 3, self.probe_distance + ds.abs(), 0.0
        )
        recovered = (
            (self.mode == 3)
            & fit
            & (self.probe_distance > self.loss_distance + 0.04)
            & (self.error.abs() < 0.006)
            & (wrap(self.line_yaw - self.yaw).abs() < 0.15)
            & (self.yaw_rate.abs() < 0.2)
        )
        self.mode[recovered] = 0
        renew = fit & (span > 2 * spacing)
        self.anchor[renew] = point[renew]
        # Save a point ON the line and its tangent while the fit is trustworthy.
        # The previous bookmark leaves a margin before the bend that caused loss.
        bookmark = (
            fit
            & (self.error.abs() < 0.012)
            & (self.mode == 0)
            & (torch.linalg.vector_norm(point - self.bookmark, dim=1) > 0.02)
        )
        self.retreat[bookmark] = self.bookmark[bookmark]
        self.retreat_yaw[bookmark] = self.bookmark_yaw[bookmark]
        self.bookmark[bookmark] = point[bookmark]
        self.bookmark_yaw[bookmark] = self.line_yaw[bookmark]
        lost = (
            (~visible | (self.error.abs() > 0.038) | failed_crossing)
            & ~endpoint
            & (self.mode == 0)
            & ~self.crossing
        )
        self.loss_distance[lost] = torch.linalg.vector_norm(point - self.retreat, dim=1)[
            lost
        ]
        self.mode[lost] = 1
        # Back along the frozen virtual line, settle, then approach the bookmark.
        # Never estimate a new tangent from observations gathered while retreating.
        tangent = torch.stack((self.retreat_yaw.cos(), self.retreat_yaw.sin()), dim=1)
        normal = torch.stack((-tangent[:, 1], tangent[:, 0]), dim=1)
        displacement = self.position - self.retreat
        along_bookmark = (displacement * tangent).sum(dim=1) * self.direction
        virtual_error = -(displacement * normal).sum(dim=1)
        virtual_heading = wrap(self.retreat_yaw - self.yaw)
        self.mode[
            (self.mode == 1) & (along_bookmark < -0.035) & (virtual_error.abs() < 0.003)
        ] = 2
        quiet = (
            (virtual_heading.abs() < 0.06)
            & (self.yaw_rate.abs() < 0.08)
            & (self.speed.abs() < 0.01)
        )
        self.settled = torch.where((self.mode == 2) & quiet, self.settled + dt, 0.0)
        approach = (self.mode == 2) & (self.settled > 0.2)
        # Measure probe travel from the bookmark, including the distance backed up.
        self.probe_distance[approach] = along_bookmark[approach]
        self.mode[approach] = 3
        virtual = (
            (self.mode == 1) | (self.mode == 2) | ((self.mode == 3) & ~self.probe_started)
        )
        reenter = (self.mode == 3) & self.probe_started
        self.line_yaw[virtual] = self.retreat_yaw[virtual]
        self.anchor[virtual] = point[virtual]
        self.heading = wrap(self.line_yaw - self.yaw)
        steering_error = torch.where(
            virtual,
            (virtual_error / virtual_heading.cos().clamp(min=0.5)).clamp(-0.04, 0.04),
            self.error,
        )
        steering_error = torch.where(
            self.crossing, predicted.clamp(-0.04, 0.04), steering_error
        )
        drive = torch.where(self.mode == 1, -self.direction, self.direction)
        self.caution = torch.where(self.mode != 0, 1.5, (self.caution - dt).clamp(min=0))
        # Repeated failure after a fresh approach is a new retreat, not a fit to
        # whatever paint happened to be visible during the backwards movement.
        self.mode[reenter & ~visible] = 1

        # Gentle lateral convergence, with gyro feedback in the inner yaw loop.
        # No heading integral: a stale tangent must not accumulate a turn demand.
        # Limit lateral convergence relative to travel along the estimated line.
        # The larger lookahead makes the approach gentler near the centre.
        correction = torch.atan(drive * steering_error / self.settings["lookahead"]).clamp(
            -0.5, 0.5
        )
        correction = torch.where(self.mode == 2, 0.0, correction)
        heading_error = self.heading + correction
        target_yaw = (self.settings["heading_gain"] * heading_error).clamp(-1.2, 1.2)
        target_yaw = torch.where(~visible & endpoint, 0.0, target_yaw)
        self.target_yaw += (target_yaw - self.target_yaw).clamp(
            -12 * dt.clamp(min=0.02), 12 * dt.clamp(min=0.02)
        )
        along = (
            drive
            * self.settings["speed"]
            / (1 + (self.heading / 0.3).square())
            / (1 + (steering_error / 0.015).square())
        )
        along /= 1 + self.yaw_rate.abs()
        along /= (weight.sum(dim=1) / 1.75).clamp(min=1).square()
        along = torch.where(self.caution > 0, along.clamp(-0.025, 0.025), along)
        along = torch.where(
            (self.mode == 2) | ((self.mode != 0) & (heading_error.abs() > 0.15)), 0.0, along
        )
        # Rotate approximately about the current intersection with the line.
        target_speed = along * self.heading.cos() + self.yaw_rate * steering_error
        # During recovery the frozen line may be outside the sensor. Steering
        # then rotates the axle; it must not translate about an unseen intersection.
        target_speed = torch.where(
            virtual, drive * 0.04 * self.heading.cos().clamp(min=0), target_speed
        )
        target_speed = torch.where(
            virtual & ((self.mode == 2) | (heading_error.abs() > 0.15)), 0.0, target_speed
        )
        target_speed = torch.where(
            self.crossing, drive * 0.04 * self.heading.cos().clamp(min=0), target_speed
        )
        # Coloured paint is still line geometry. Keep following its bends;
        # only the short passage beyond the physical endpoint is blind travel.
        target_speed = torch.where(endpoint, target_speed.clamp(-0.06, 0.06), target_speed)
        target_speed = torch.where(endpoint & ~visible, self.direction * 0.06, target_speed)
        self.target_speed += (target_speed - self.target_speed).clamp(
            -0.7 * dt.clamp(min=0.02), 0.7 * dt.clamp(min=0.02)
        )
        # PI speed feedback overcomes sphere drag; yaw feedback damps steering.
        speed_error = self.target_speed - self.speed
        self.speed_integral = (self.speed_integral + dt * speed_error).clamp(-0.08, 0.08)
        throttle = 0.8 * self.target_speed + 0.6 * speed_error + 0.5 * self.speed_integral
        steering = 0.035 * self.target_yaw + 0.1 * (self.target_yaw - self.yaw_rate)
        self.command = torch.stack((throttle - steering, throttle + steering), dim=1).clamp(
            -0.5, 0.5
        )
        self.previous_time, self.previous_encoders = time.clone(), encoders.clone()
        self.visible = visible
        return self.command.clone()

    def debug(self, index=0):
        if self.position is None:
            return {}
        return {
            "line_offset_m": float(self.error[index]),
            "line_visible": bool(self.visible[index]),
            "crossing": bool(self.crossing[index]),
            "drive_direction": float(self.direction[index]),
            "heading_error_rad": float(self.heading[index]),
            "recovering": int(
                self.mode[index]
            ),  # 0: follow, 1: retreat, 2: settle, 3: approach
            "speed_m_s": float(self.speed[index]),
            "yaw_rate": float(self.yaw_rate[index]),
            "target_yaw_rate": float(self.target_yaw[index]),
            "command": self.command[index].detach().cpu().numpy().copy(),
        }
