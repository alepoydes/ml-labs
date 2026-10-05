# Line Following

Train a controller for a simulated robot that follows a line to a marked endpoint
and returns to the start as quickly as possible. The whole journey
is timed, including both directions, the turnaround, stops, and recovery from
mistakes. The robot must remain upright and cover every part of the route in
order.

The public dynamics are described in the [mechanics guide](dynamics.md). Prepared
practice courses, a batched environment, and a slow baseline are available.
The current settings are a pilot profile; controller performance and the full
parameter family still need calibration before the assessed release. The local
viewer and isolated instructor evaluation use this same public environment.

## What you will learn

The task combines feedback control, learning from interaction, adaptation to
unknown dynamics, and memory. You must choose how fast to travel while following
the line, maintaining balance, and recovering from mistakes. The observations
are local, so their history can help you estimate direction and motion.

You may combine conventional control with learned components, train a policy
directly, or learn a model for use by a controller. Your solution must include
training or fitting, and you should explain what it learns and measure the
effect. Explicitly identifying every physical parameter is not required.

## The robot

The robot has two independently driven wheels on a common axle. A passive sphere
behind the wheels provides a third point of support. It can rotate in any
direction, with viscous resistance relative to the robot body: faster relative
rotation produces a larger resisting torque.

When leaning slightly backward, the robot can rest and drive on its wheels and
the sphere. Near the upright balancing position, the sphere lifts clear of the
floor, removing the resistance transmitted through that contact. This can make
faster travel worthwhile, but the controller must then maintain balance.

Using the sphere, partly unloading it, lifting it, and touching down again are
all allowed. There is no separate requirement or bonus for keeping it airborne.

All three floor contacts can slip both along and across the direction of travel.
Traction is limited and depends on the load at each contact. Acceleration,
braking, and turning share the available traction. Motors also have finite
capabilities. Wheel rotation measured by an encoder may differ from actual
movement over the ground.

The physical model and pilot parameter ranges are public. The actual
parameters are chosen at the start of each episode and are not supplied to your
controller. They stay unchanged for the complete outward and return journey.
You can adapt using the robot's observed response to your actions.

## Observations and control

Your controller receives one transverse row of 16 colour measurements from
under the chassis, together with gyroscope, accelerometer, and wheel-encoder
readings. Each optical sector reports the average RGB colour over its sensing
area. The line has a fixed physical width, intended to span approximately two
sectors in the nominal sensor pose.

The row sees only its designated area beneath the chassis. It does not provide
a forward view or a map. You can use the history of observations and actions to
estimate the local direction of the line, motion, and dynamics.

The controller commands the left and right drives. Its memory and internal
estimates persist through the turnaround and return journey, and are reset
between independent episodes. Sensor/action units, rates, and pilot limits appear
below with the simulator API.

During evaluation, the controller is not told its true position, distance along
the route, sampled physical parameters, which branch the judge is tracking, or
whether a movement earned progress. Endpoint colours are ordinary physical
observations; they do not indicate that the preceding route was accepted.

## Courses

The ground is flat. Courses can contain straight sections, smooth bends, sharp
corners, sections that pass close to one another, and self-intersections.

At a crossing, follow the smooth continuation of the branch on which you arrived.
The crossing angle is sufficiently large to distinguish the branches, and both
branches are smooth in a neighbourhood of the crossing. Sharp corners are kept
outside that neighbourhood. The pilot requires crossing angles of at least 50°
and keeps marked corners at least 0.2 m from crossings.

Distinct coloured sections of the line mark the start and the far endpoint.
At the far endpoint, follow the same route back. Driving forward or backward is
allowed on either leg; rotating the body is optional. The
colours identify endpoints, not directions at crossings or future manoeuvres.

You will receive several prepared practice courses with varied geometry. Hidden
evaluation courses will differ from them and may combine permitted situations
in ways absent from the practice collection. All courses obey the same published
rules. Expect to generalise beyond memorising the practice routes.

## Completing the route

Every part of the line must pass under the designated chassis footprint in route
order. The centre of the robot does not have to remain exactly over the centre
of the line. The judge follows the continuation of the current route segment,
including at crossings and near other branches.

Coverage must remain continuous throughout movement, including between sensor
updates. Being correctly placed at the beginning and end of an update interval
does not excuse skipping part of the route in between. A close or intersecting
branch cannot replace the required continuation.

Losing the line is recoverable. Your accepted progress freezes when tracking is
lost. Returning to the already accepted part lets you resume; to earn further
progress you must reach the previous frontier and follow the remaining line.
Rejoining ahead of that frontier does not earn progress, even if you keep driving
forward from there.

For example, if you leave after 10 metres and rejoin at 12 metres, those two metres
remain missing. You must return to 10 metres or earlier and follow the missing
section. Returning to an earlier point does not erase the progress already earned.
The same rule applies independently on the return journey.

Seeing an endpoint marker is insufficient if part of that leg was skipped.
The judge does not tell the controller whether individual movements were accepted.

## Timing and failure

There is one uninterrupted timed episode. The first leg is not a free scouting
run. The controller may use it to learn the route and dynamics, but that time
counts toward the same total as the return leg. All searching, stopping, and
turning time counts too.

Falling fails the episode and ends simulation immediately. Ordinary contact with
the support sphere is allowed. Falling forward or sideways is not a recovery
technique. The pilot fall boundary and episode time limit are specified below.

For a completed journey taking simulated time $T$, the episode score is

$$
S = \min\left(100,\;100\frac{T_{100}}{T}\right).
$$

Here $T_{100}$ is a mechanical reference time for the course and sampled robot.
It uses full knowledge of the route and dynamics to estimate fast travel along
the line's centre, accounting for speed and acceleration limits, curvature,
both directions, and the turnaround. It includes no cost for discovering the
route or identifying the robot. The public `reference_time` function implements
this approximation. Its settings will be frozen for the assessed version.

The reference is an approximate normalisation. Legal corner cutting within the
coverage rule or useful sliding may beat it; the score is capped at 100. Taking
$1.67\,T_{100}$ earns approximately 60 points; taking $2.5\,T_{100}$ earns 40.

A fall or failure to complete the journey before the episode time limit earns
zero. Partial progress earns no points. The final score is the mean over all
scheduled evaluation episodes, including failed episodes. Thus faster successful
journeys can compensate for some failures. Simulator and infrastructure failures
are reported separately; an interrupted evaluation does not yield an official
aggregate. Academic grade thresholds remain to be defined before assessment.

## Training and reporting

You may repeat practice episodes with different permitted physical parameters,
use their outcomes for learning, and compare different controllers. Training
rewards can guide optimisation, but accepted progress and rewards are not part
of the acting controller's evaluation observations. Within an episode, adaptation
must use its permitted observations and action history.

A simple supported reference controller is provided as `baseline`. It follows
the optical centroid slowly, uses elementary line recovery, and returns in
reverse. Its preparation function saves hand-tuned settings; it is not a learned
solution. In a four-episode probe on the spline profile it completed two straight courses
and one smooth course; the other smooth case timed out at the 60-second pilot
cutoff. This does not establish performance on sharp corners, crossings, or the
complete benchmark. Scores measured on the earlier polyline profile are obsolete. You should measure your
own completion rate and journey time, including recovery costs, and compare
performance on familiar and unfamiliar courses. Separate outward and return
times can show whether experience from the first leg helps.

Submit a self-contained solution with reproducible training, inference and all
required artifacts, an explanation of the learned component and experiments,
and an honest declaration of AI assistance. Final inference and adaptation run
on CPU without network access, within limits to be published before release.

## Running and writing a solution

From the student project root:

```sh
./step_0_update_resources
./step_1_new_solution line_following my-controller --template baseline
./step_2_train work/line_following/my-controller
./step_3_evaluate work/line_following/my-controller
./step_4_pack work/line_following/my-controller
./view_lab line_following --solution work/line_following/my-controller
```

The small course collection is bundled with this pilot and copied into `data/`
with the same size/hash checks as downloaded resources. Use the `.cmd` scripts
on Windows. Replace the baseline's preparation procedure with your own learning
or fitting. `starter` demonstrates the smallest interface and commands zero
motor effort.

The viewer supports pause, one physical step, one controller interval, speed control,
batch-row selection, and rewind of recorded history. Resuming returns to the live
simulation state. Optional `debug(index=0)` diagnostics are copied after each action;
see [graphical inspection](../../docs/development.md#graphical-inspection-and-manual-play).

The constructor is `Solution(artifact_dir, config, seed)`. Inference is
`act(observation, reset)`: observations are float32 PyTorch tensors with a leading
batch dimension, even for one robot. Return float32 commands `(B, 2)` in `[-1, 1]`.
Positive equal commands drive forward. The Boolean mask `reset` has shape `(B,)`;
clear the indicated rows of controller memory before processing their inputs.
`act` runs with autograd available in both local and isolated evaluation, so online
adaptation may use `backward()`. Use your own `torch.no_grad()` for inference that
does not need gradients.
Memory persists across the far endpoint. Reset must restore the episode's initial
controller state, including any within-episode adaptation.

| Observation | Shape | Meaning |
| --- | --- | --- |
| `time` | `(B,)` | Simulated seconds since the episode began |
| `rgb` | `(B, 16, 3)` | RGB averages in `[0, 1]`, sectors ordered from body right to left |
| `gyro` | `(B, 3)` | Body-frame angular velocity in rad/s |
| `accel` | `(B, 3)` | Body-frame specific acceleration at the axle centre in m/s² |
| `encoders` | `(B, 2)` | Left/right wheel angles relative to the body in radians |

The current CPU environment uses float64 physics at 1 ms and observations/actions
every 20 ms. The first compiled step includes compiler startup; reuse an environment
and its batch size. Use `compiled=False` for ordinary Python debugging. Never wrap
the simulator in `torch.inference_mode()`; its kinematics uses differentiation.

```python
from pathlib import Path
from coursekit.labs.line_following_env import Environment
from coursekit.labs.line_following_task import benchmark
from coursekit.training import load_solution

cases = benchmark(Path("data/line_following"), "train")
env = Environment([c for c, seed in cases], [seed for c, seed in cases])
controller = load_solution(Path("work/line_following/my-controller"))
observation = env.observe()
action = controller.act(observation, env.first)
observation, reward, terminated, truncated, info = env.step(action)
done = terminated | truncated
if done.any():
    observation = env.reset(done)
```

`step` holds each action for one control interval and freezes finished rows.
`terminated` identifies completion or falling; `truncated` identifies the time
limit, which scores zero. The final observation is retained until explicit reset.
`reset(mask)` samples a new episode only for the selected rows. Supply `courses`
and `seeds` for those rows to reproduce particular episodes. Environment randomness
is independent between rows and from policy randomness.

The default training reward is newly accepted distance divided by the full
out-and-back distance. It is zero on retraced ground. Construct your own reward
using `info` if useful. `info` contains true pose, progress, leg, outcome, timing,
reference time, and score; it is never passed to evaluated `act` calls.

## Pilot geometry and parameter settings

The canonical course is an ordered cubic Bezier spline, stored as control points,
parameter knots, and explicit corner flags. Smooth joins are C2; genuine corners
keep their tangent discontinuity. `Course.at(s)` evaluates the spline at arc length
`s`; `Course.at(s, derivative=n)` evaluates the derivative with respect to the
spline parameter defined by its knots. `Course.geometry(s)` returns curvature
and its arc-length derivative.
Both come from analytic polynomial derivatives. Arc length uses 16-point Gaussian
quadrature, checked against 8-point quadrature during adaptive subdivision.

All geometry consumers share that spline. Cached chords have a certified pointwise
error of at most 0.01 mm. The judge erodes clipping halfplanes by this error before
accepting a curve interval. Optical projection evaluates the spline itself,
initialized by these chords; its distance error is bounded by 0.02 mm. Rendering
uses the same subdivision. Optical sectors average 4×4 samples
over rectangles in one transverse row, 128 mm wide and 8 mm deep. Line width is
16 mm. The first and last 120 mm of line are blue and red. The chassis footprint
is the ground projection of a body-fixed 140×128 mm rectangle at the axle centre.
Coverage uses a conservative inner approximation to its swept region at each
physical step, including the endpoint footprints and a certified translation
sweep, with the additional spline-error erosion. It never substitutes the convex hull of two rotated footprints. See the
small public `line_following_course` module for the calculation.

The reference profile starts at rest, stops to reverse travel at the far endpoint,
and allows a moving finish. Marked sharp corners include a stop and an in-place
change of heading; independent drive, traction, and yaw limits give an optimistic
speed profile on the remaining line. The reference grid has at most 1 mm arc-length
spacing and includes genuine polynomial joins with limits from both sides.
Exact subdivisions do not introduce artificial reference constraints. Speed passes
use cumulative acceleration bounds; time is integrated along the spline's arc
coordinate. Legal driving may beat this approximation.

Course archives and robot recordings use format version 2. Earlier polyline files
are rejected explicitly; update resources and create new robot recordings. Existing
controller observations and actions retain their API.

`sample_robot` publishes the complete pilot distribution. Component masses and
their inertias scale together by 0.85–1.15; body centre-of-mass height scales by
0.9–1.1 with consistent inertia scaling. Wheel friction lies in 0.65–0.95, support
friction in 0.45–0.75, motor stall torque in 0.065–0.095 N m, no-load speed in
38–52 rad/s, and sphere damping in 0.00015–0.00025 N m s. Geometry and contact
stiffness use the nominal mechanics profile. Independent zero-mean Gaussian
sensor noise has standard deviations 0.008 for RGB, 0.004 rad/s for gyro,
0.025 m/s² for acceleration, and 0.0005 rad for encoders. RGB is clipped to `[0, 1]`.
The pilot time limit is 120 simulated seconds; a body-up tilt of 0.9 rad is a fall.
