"""Batched line-following episodes over the shared public mechanics.

The policy sees copied float32 observations, never judge state. Local training can
inspect info and choose its own reward. reset(mask) changes only selected rows;
terminal rows remain frozen until explicitly reset. Ordinary torch.no_grad is
compatible with the mechanics; torch.inference_mode is not.
"""

from collections.abc import Callable, Sequence
from functools import lru_cache

import numpy as np
import torch
from torch.func import vmap

from coursekit.common import CourseError
from coursekit.labs import line_following as physics
from coursekit.labs.line_following_course import Course, advance, optical, reference_time

PHYSICS_DT = 0.001
CONTROL_STEPS = 20
TIME_LIMIT = 120.0
OBSERVATION_SHAPES = {
    "time": (),
    "rgb": (16, 3),
    "gyro": (3,),
    "accel": (3,),
    "encoders": (2,),
}


def sample_robot(rng: np.random.Generator) -> physics.Robot:
    """Pilot parameter family; shape is fixed and inertias scale with masses."""
    model = physics.robot()
    scale = torch.from_numpy(rng.uniform(0.85, 1.15, 4))
    height = rng.uniform(0.9, 1.1)
    inertia = model.inertia * scale[:, None]
    inertia[0] *= height**2
    offset = model.offset.clone()
    offset[0, 2] *= height
    return model._replace(
        mass=model.mass * scale,
        inertia=inertia,
        offset=offset,
        friction=torch.from_numpy(
            np.r_[rng.uniform(0.65, 0.95, 2), rng.uniform(0.45, 0.75)]
        ),
        stall=torch.from_numpy(rng.uniform(0.065, 0.095, 2)),
        speed=torch.from_numpy(rng.uniform(38.0, 52.0, 2)),
        ball_drag=torch.tensor(rng.uniform(0.00015, 0.00025), dtype=torch.float64),
    )


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


@lru_cache(maxsize=1)
def _compiled_step():
    return physics.compile_step(batched=True, model_batched=True)


def _seeds(values, size):
    values = np.asarray(values)
    if (
        values.shape != (size,)
        or values.dtype.kind not in "iu"
        or (values < 0).any()
        or (values.dtype.kind == "u" and values.max(initial=0) > np.iinfo(np.int64).max)
    ):
        raise CourseError("Provide one nonnegative 63-bit integer seed per course.")
    return values.astype(np.int64, copy=True)


class Environment:
    def __init__(
        self,
        courses: Sequence[Course],
        seeds: Sequence[int] | None = None,
        *,
        compiled=True,
        time_limit=TIME_LIMIT,
        noise=True,
    ):
        if not courses or any(course.length < 0.3 for course in courses):
            raise CourseError(
                "Episodes need at least one course, each at least 0.3 m long."
            )
        if not np.isfinite(time_limit) or time_limit <= 0:
            raise CourseError("The episode time limit must be positive and finite.")
        self.courses = list(courses)
        self.reverse = [c.reversed() for c in courses]
        self.size = len(courses)
        self.seeds = _seeds(range(self.size) if seeds is None else seeds, self.size)
        self.episodes = np.full(self.size, -1, dtype=np.int64)
        self.rng = [None] * self.size
        self.noise = noise
        self.limit_steps = int(np.ceil(time_limit / PHYSICS_DT))
        self._advance = _compiled_step() if compiled else vmap(physics.step)
        nominal = physics.robot()
        state = physics.initial_state(nominal)
        self.model = physics.Robot(
            *(v.expand(self.size, *v.shape).clone() for v in nominal)
        )
        self.state = physics.State(*(v.expand(self.size, *v.shape).clone() for v in state))
        self.ticks = np.zeros(self.size, dtype=np.int64)
        self.leg = np.zeros(self.size, dtype=np.int64)
        self.frontier = np.zeros(self.size)
        self.outward_time = np.full(self.size, np.nan)
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

    def reset(self, mask=None, *, courses=None, seeds=None):
        mask = np.ones(self.size, dtype=bool) if mask is None else np.asarray(mask)
        if mask.shape != (self.size,) or mask.dtype != np.bool_:
            raise CourseError("Reset mask must be Boolean with shape (B,).")
        indices = np.flatnonzero(mask)
        if courses is not None and (
            len(courses) != len(indices) or any(c.length < 0.3 for c in courses)
        ):
            raise CourseError("Provide one course of at least 0.3 m per reset row.")
        if seeds is not None:
            seeds = _seeds(seeds, len(indices))
        for position, i in enumerate(indices):
            if courses is not None:
                self.courses[i] = courses[position]
                self.reverse[i] = courses[position].reversed()
            if seeds is not None:
                self.seeds[i] = seeds[position]
                self.episodes[i] = -1
            self.episodes[i] += 1
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seeds[i], self.episodes[i]])
            )
            self.rng[i] = rng
            model = sample_robot(rng)
            state = physics.initial_state(model)
            direction = self.courses[i].at(0.0, derivative=1)
            state.q[:2] = torch.from_numpy(self.courses[i].at(0.0).copy())
            state.q[5] = float(np.arctan2(direction[1], direction[0]))
            for target, value in zip(self.model, model):
                target[i] = value
            for target, value in zip(self.state, state):
                target[i] = value
            self.ticks[i] = self.leg[i] = 0
            self.status[i] = "running"
            self.outward_time[i] = np.nan
            self.reference[i] = reference_time(self.courses[i], model)
            q = state.q.numpy()
            r = physics.rotation(state.q).numpy()
            self.frontier[i] = advance(self.courses[i], 0.0, q, q, r, r)
        self._first |= torch.from_numpy(mask.copy())
        self._sense(mask, np.zeros(self.size))
        return self.observe()

    def _sense(self, mask, elapsed):
        velocity = vmap(physics.velocity)(self.state, self.model)
        _, angular, rotation = vmap(physics.motion)(self.state.q, velocity, self.model)
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
                "rgb": optical(self.courses[i], q[:3], r),
                "gyro": local_angular[i].numpy(),
                "accel": local_acceleration[i].numpy(),
                "encoders": q[6:].copy(),
            }
            for name, sigma in (
                ("rgb", 0.008),
                ("gyro", 0.004),
                ("accel", 0.025),
                ("encoders", 0.0005),
            ):
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
        action = action.detach().to(dtype=torch.float64, device="cpu")
        beginning, old_progress = self.ticks.copy(), self.progress.copy()
        was_active = self.status == "running"
        self._first.zero_()
        r0 = vmap(physics.rotation)(self.state.q).numpy()
        for _ in range(CONTROL_STEPS):
            active = self.status == "running"
            if not active.any():
                break
            previous = self.state
            mask = torch.from_numpy(active)
            # Reuse the engine's terminal masking also for completed/timeout rows.
            stepping = previous._replace(fallen=previous.fallen | ~mask)
            moved, diagnostics = self._advance(stepping, action, self.model)
            physics.check(moved, diagnostics)
            self.state = physics.State(
                *(
                    torch.where(mask.reshape(self.size, *([1] * (a.ndim - 1))), a, b)
                    for a, b in zip(moved, previous)
                )
            )
            r1 = vmap(physics.rotation)(self.state.q).numpy()
            self.ticks[active] += 1
            for i in np.flatnonzero(active):
                if self.state.fallen[i]:
                    self.status[i] = "fallen"
                    continue
                course = self.courses[i] if self.leg[i] == 0 else self.reverse[i]
                q0, q1 = previous.q[i].numpy(), self.state.q[i].numpy()
                self.frontier[i] = advance(course, self.frontier[i], q0, q1, r0[i], r1[i])
                if self.frontier[i] >= course.length:
                    if self.leg[i] == 0:
                        self.leg[i] = 1
                        self.outward_time[i] = self.ticks[i] * PHYSICS_DT
                        # The endpoint pose starts the new leg; do not reuse earlier sweep.
                        self.frontier[i] = advance(
                            self.reverse[i], 0.0, q1, q1, r1[i], r1[i]
                        )
                    else:
                        self.status[i] = "finished"
                if self.status[i] == "running" and self.ticks[i] >= self.limit_steps:
                    self.status[i] = "timeout"
            r0 = r1
            if on_step is not None:
                on_step(self)
        elapsed = (self.ticks - beginning) * PHYSICS_DT
        self._sense(was_active, elapsed)
        length = np.array([c.length for c in self.courses])
        reward = torch.from_numpy(
            ((self.progress - old_progress) / (2 * length)).astype(np.float32)
        )
        terminated = torch.from_numpy(np.isin(self.status, ["finished", "fallen"]))
        truncated = torch.from_numpy(self.status == "timeout")
        return self.observe(), reward, terminated, truncated, self.info()

    def info(self):
        elapsed = self.ticks * PHYSICS_DT
        score = np.where(
            self.status == "finished",
            np.minimum(100.0, 100 * self.reference / np.maximum(elapsed, PHYSICS_DT)),
            0.0,
        )
        return {
            "time": elapsed.copy(),
            "progress": self.progress.copy(),
            "leg": self.leg.copy(),
            "status": self.status.copy(),
            "score": score,
            "reference_time": self.reference.copy(),
            "outward_time": self.outward_time.copy(),
            "q": self.state.q.numpy().copy(),
        }
