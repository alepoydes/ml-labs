"""Batched line-following episodes over the shared public mechanics.

The policy sees copied float32 observations, never judge state. Local training can
inspect info and choose its own reward. reset(mask) changes only selected rows;
terminal rows remain frozen until explicitly reset. Ordinary torch.no_grad is
compatible with the mechanics; torch.inference_mode is not.
"""

from collections.abc import Callable, Sequence

import numba
import numpy as np
import torch
from torch.func import vmap

from coursekit.common import CourseError
from coursekit.labs import line_following as physics
from coursekit.labs import line_following_geometry as geometry
from coursekit.labs import line_following_rollout as rollout
from coursekit.labs.line_following_cases import SENSOR_NAMES, Case
from coursekit.labs.line_following_course import advance, optical, reference_time
from coursekit.labs.line_following_results import SCORE_TIME_FACTOR

PHYSICS_DT = 0.001
CONTROL_STEPS = 20
TIME_LIMIT_FACTOR = 30.0
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


def _kinematics(state, model):
    velocity = physics.velocity(state, model)
    _, angular, rotation = physics.motion(state.q, velocity, model)
    return velocity, angular, rotation


def _inspect(state, diagnostics):
    return (
        state,
        vmap(physics.rotation)(state.q),
        physics.numerical_status(state, diagnostics),
    )


def _step(state, action, model):
    return _inspect(*vmap(physics.step)(state, action, model))


class Environment:
    def __init__(
        self,
        cases: Sequence[Case],
        *,
        compiled=True,
        time_limit=None,
        noise=True,
    ):
        if not cases or any(not isinstance(case, Case) for case in cases):
            raise CourseError("Provide at least one fixed Case.")
        if time_limit is not None and (not np.isfinite(time_limit) or time_limit <= 0):
            raise CourseError("The episode time limit must be positive and finite.")
        self.cases = list(cases)
        self.courses = [c.course for c in cases]
        self.reverse = [c.reversed() for c in self.courses]
        self.size = len(cases)
        self.rng = [None] * self.size
        self.noise = noise
        self.time_limit_override = time_limit
        self.limit_steps = np.zeros(self.size, dtype=np.int64)
        self.compiled = compiled
        self._advance = _step
        self._kinematics = vmap(_kinematics)
        nominal = physics.robot()
        state = physics.initial_state(nominal)
        self.model = physics.Robot(*(v.repeat(self.size, *([1] * v.ndim)) for v in nominal))
        # repeat materializes normal strides even for a one-element scalar batch.
        self.state = physics.State(*(v.repeat(self.size, *([1] * v.ndim)) for v in state))
        self.ticks = np.zeros(self.size, dtype=np.int64)
        self.leg = np.zeros(self.size, dtype=np.int64)
        self.frontier = np.zeros(self.size)
        self.outward_time = np.full(self.size, np.nan)
        self.travelled_distance = np.zeros(self.size)
        self.status = np.full(self.size, "running", dtype="U8")
        self.reference = np.zeros(self.size)
        self._first = torch.ones(self.size, dtype=torch.bool)
        self._observation = {
            key: torch.zeros((self.size, *shape), dtype=torch.float32)
            for key, shape in OBSERVATION_SHAPES.items()
        }
        self._linear = torch.zeros((self.size, 3), dtype=torch.float64)
        self.reset()

    @property
    def first(self):
        return self._first.clone()

    @property
    def done(self):
        return torch.from_numpy(self.status != "running")

    @property
    def progress(self):
        return self.frontier + self.leg * np.array([c.length for c in self.courses])

    def observe(self):
        return {key: value.clone() for key, value in self._observation.items()}

    def reset(self, mask=None, *, cases=None):
        mask = np.ones(self.size, dtype=bool) if mask is None else np.asarray(mask)
        if mask.shape != (self.size,) or mask.dtype != np.bool_:
            raise CourseError("Reset mask must be Boolean with shape (B,).")
        indices = np.flatnonzero(mask)
        if cases is not None and (
            len(cases) != len(indices) or any(not isinstance(c, Case) for c in cases)
        ):
            raise CourseError("Provide one fixed Case per reset row.")
        for position, i in enumerate(indices):
            if cases is not None:
                self.cases[i] = cases[position]
                self.courses[i] = cases[position].course
                self.reverse[i] = self.courses[i].reversed()
            case = self.cases[i]
            self.rng[i] = case.noise_generator()
            model, state = case.robot, case.initial
            for target, value in zip(self.model, model):
                target[i] = value
            for target, value in zip(self.state, state):
                target[i] = value
            self.ticks[i] = self.leg[i] = 0
            self.status[i] = "running"
            self.outward_time[i] = np.nan
            self.travelled_distance[i] = 0.0
            self.reference[i] = reference_time(self.courses[i], model)
            limit = (
                TIME_LIMIT_FACTOR * self.reference[i]
                if self.time_limit_override is None
                else self.time_limit_override
            )
            self.limit_steps[i] = int(np.ceil(limit / PHYSICS_DT))
            q = state.q.numpy()
            r = physics.rotation(state.q).numpy()
            self.frontier[i] = advance(self.courses[i], 0.0, q, q, r, r)
        if self.compiled:
            # Views share the parameter storage; resets update those same bytes.
            self._models = physics.Robot(*(v.numpy() for v in self.model))
            self._courses = geometry.pack(self.courses)
            self._reverse = geometry.pack(self.reverse)
        self._first |= torch.from_numpy(mask.copy())
        self._sense(mask, np.zeros(self.size))
        return self.observe()

    @torch.no_grad()
    def _sense(self, mask, elapsed):
        if self.compiled:
            velocity, angular, rotation = (
                torch.from_numpy(v)
                for v in rollout.kinematics(
                    self.state.q.numpy(), self.state.p.numpy(), self._models
                )
            )
        else:
            velocity, angular, rotation = self._kinematics(self.state, self.model)
        acceleration = (velocity[:, :3] - self._linear) / torch.from_numpy(
            np.maximum(elapsed, PHYSICS_DT)
        )[:, None]
        acceleration[torch.from_numpy(elapsed == 0)] = 0
        acceleration[:, 2] += self.model.gravity
        local_acceleration = torch.einsum("bij,bi->bj", rotation, acceleration)
        local_angular = torch.einsum("bij,bi->bj", rotation, angular)
        for i in np.flatnonzero(mask):
            rng = self.rng[i]
            q, r = self.state.q[i].numpy(), rotation[i].numpy()
            readings = {
                "time": np.array(self.ticks[i] * PHYSICS_DT),
                "rgb": (
                    geometry.optical(geometry.row(self._courses, i), q, r)
                    if self.compiled
                    else optical(self.courses[i], q[:3], r)
                ),
                "gyro": local_angular[i].numpy(),
                "accel": local_acceleration[i].numpy(),
                "encoders": q[6:].copy(),
            }
            for name, sigma in zip(SENSOR_NAMES, self.cases[i].noise_std):
                if self.noise:
                    readings[name] = readings[name] + rng.normal(
                        0.0, sigma, readings[name].shape
                    )
            readings["rgb"] = np.clip(readings["rgb"], 0, 1)
            for name, value in readings.items():
                self._observation[name][i] = torch.as_tensor(value, dtype=torch.float32)
            self._linear[i] = velocity[i, :3]

    @torch.no_grad()
    def step(self, action: torch.Tensor, *, on_step: Callable | None = None):
        validate_action(action, self.size)
        # Controller views must reuse the layout compiled during engine warmup.
        action = action.detach().to(
            dtype=torch.float64, device="cpu", memory_format=torch.contiguous_format
        )
        beginning, old_progress = self.ticks.copy(), self.progress.copy()
        was_active = self.status == "running"
        self._first.zero_()
        if self.compiled:
            self._native_block(action.numpy(), on_step)
        else:
            self._reference_block(action, on_step)
        elapsed = (self.ticks - beginning) * PHYSICS_DT
        self._sense(was_active, elapsed)
        length = np.array([c.length for c in self.courses])
        reward = torch.from_numpy(
            ((self.progress - old_progress) / (2 * length)).astype(np.float32)
        )
        terminated = torch.from_numpy(np.isin(self.status, ["finished", "fallen"]))
        truncated = torch.from_numpy(self.status == "timeout")
        return self.observe(), reward, terminated, truncated, self.info()

    def _install_native(self, q, p, spin, fallen, counters):
        self.state = physics.State(
            *(torch.from_numpy(np.ascontiguousarray(v)) for v in (q, p, spin, fallen))
        )
        self.ticks[:] = counters[:, 0]
        self.leg[:] = counters[:, 1]
        self.frontier[:] = counters[:, 2]
        self.outward_time[:] = counters[:, 3]
        self.travelled_distance[:] = counters[:, 4]
        self.status[:] = np.array(["running", "finished", "fallen", "timeout"])[
            counters[:, 5].astype(np.int64)
        ]

    def _native_block(self, action, on_step):
        # Commit only a numerically valid block. Each row stops at its own terminal
        # step; optional snapshots let the viewer pace/replay every physical frame.
        q, p, spin, fallen = (v.numpy().copy() for v in self.state)
        statuses = np.array(["running", "finished", "fallen", "timeout"])
        codes = (self.status[:, None] == statuses).argmax(axis=1)
        counters = np.column_stack(
            (
                self.ticks,
                self.leg,
                self.frontier,
                self.outward_time,
                self.travelled_distance,
                codes,
            )
        )
        beginning = self.ticks.copy()
        steps = CONTROL_STEPS
        history = np.empty((self.size, steps if on_step else 0, 26))
        if on_step:
            history[:] = np.column_stack((q, p, spin, fallen, counters))[:, None, :]
        errors = np.zeros(self.size, dtype=np.int64)
        # Tiny matrix operations scale poorly with a large thread pool. Bound the
        # engine's parallelism while preserving the calling thread's Numba mask.
        threads = numba.get_num_threads()
        numba.set_num_threads(min(threads, 4))
        try:
            rollout.block(
                q,
                p,
                spin,
                fallen,
                self._models,
                self._courses,
                self._reverse,
                action,
                counters,
                self.limit_steps,
                steps,
                history,
                errors,
                PHYSICS_DT,
            )
        finally:
            numba.set_num_threads(threads)
        if errors.any():
            raise FloatingPointError("Non-finite or nonconvergent robot dynamics.")
        if on_step:
            counts = counters[:, 0].astype(np.int64) - beginning
            for row, count in enumerate(counts):
                if count:
                    history[row, count:] = history[row, count - 1]
            for i in range(int(counts.max())):
                f = history[:, i]
                self._install_native(
                    f[:, :8], f[:, 8:16], f[:, 16:19], f[:, 19].astype(bool), f[:, 20:]
                )
                on_step(self)
        else:
            self._install_native(q, p, spin, fallen, counters)

    def _reference_block(self, action, on_step):
        r0 = vmap(physics.rotation)(self.state.q).numpy()
        for _ in range(CONTROL_STEPS):
            active = self.status == "running"
            if not active.any():
                break
            previous = self.state
            mask = None if active.all() else torch.from_numpy(active)
            # Completed rows need masking; the usual all-active batch does not.
            stepping = (
                previous
                if mask is None
                else previous._replace(fallen=previous.fallen | ~mask)
            )
            moved, rotation, status = self._advance(stepping, action, self.model)
            physics.check_status(status)
            self.state = (
                moved
                if mask is None
                else physics.State(
                    *(
                        torch.where(mask.reshape(self.size, *([1] * (a.ndim - 1))), a, b)
                        for a, b in zip(moved, previous)
                    )
                )
            )
            r1 = rotation.numpy()
            self.ticks[active] += 1
            q0, q1, fallen = (
                previous.q.numpy(),
                self.state.q.numpy(),
                self.state.fallen.numpy(),
            )
            # Ground path of the axle centre, including slips and the falling step.
            # Count physical steps, independently of sensing and recording frequency.
            self.travelled_distance += np.linalg.norm(q1[:, :2] - q0[:, :2], axis=1)
            for i in np.flatnonzero(active):
                if fallen[i]:
                    self.status[i] = "fallen"
                    continue
                course = self.courses[i] if self.leg[i] == 0 else self.reverse[i]
                self.frontier[i] = advance(
                    course, self.frontier[i], q0[i], q1[i], r0[i], r1[i]
                )
                if self.frontier[i] >= course.length:
                    if self.leg[i] == 0:
                        self.leg[i] = 1
                        self.outward_time[i] = self.ticks[i] * PHYSICS_DT
                        # The endpoint pose starts the new leg; do not reuse earlier sweep.
                        self.frontier[i] = advance(
                            self.reverse[i], 0.0, q1[i], q1[i], r1[i], r1[i]
                        )
                    else:
                        self.status[i] = "finished"
                if self.status[i] == "running" and self.ticks[i] >= self.limit_steps[i]:
                    self.status[i] = "timeout"
            r0 = r1
            if on_step is not None:
                on_step(self)

    def info(self):
        elapsed = self.ticks * PHYSICS_DT
        progress = self.progress
        route_length = 2 * np.array([c.length for c in self.courses])
        score = np.where(
            self.status == "finished",
            np.minimum(
                100.0,
                100 * SCORE_TIME_FACTOR * self.reference / np.maximum(elapsed, PHYSICS_DT),
            ),
            0.0,
        )
        return {
            "time": elapsed.copy(),
            "progress": progress,
            "route_length": route_length,
            "completion_fraction": progress / route_length,
            "travelled_distance": self.travelled_distance.copy(),
            "leg": self.leg.copy(),
            "status": self.status.copy(),
            "score": score,
            "reference_time": self.reference.copy(),
            "time_limit": self.limit_steps * PHYSICS_DT,
            "outward_time": self.outward_time.copy(),
            "q": self.state.q.numpy().copy(),
        }
