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

## View the practice courses

Open the [offline course atlas](resources/index.html) to see all 65 routes together,
filter training/validation courses, and find their lengths and fixed robot seeds.
After resource preparation, the same atlas is available at
`data/line_following/index.html`. It works without Python or a network connection.

For the interactive simulator, from the student project root run:

```sh
./view_lab line_following
```

On Windows use `.\view_lab.cmd line_following`. The viewer prepares the bundled
course files automatically; browsing does not require a working solution, training
or **Start**. Choose **Course** in the always-visible row (or use
**Previous course / Next course**), then
enable **Top view** and click **Fit track**. Scroll to zoom and drag to pan. Blue
marks the start and return finish; red marks the far turnaround. The status line
shows the course name and one-way length.

The current collection has **50 training courses** (`train-01`…`train-50`) and
**15 validation courses** (`validation-01`…`validation-15`). Work through training
in order, then mix earlier and later cases to avoid forgetting the basics:

| Training courses | Suggested progression |
| --- | --- |
| 01–02 | The same one-metre straight route with two different robots; drive out and return. |
| 03–10 | Wide curves, a rounded triangle and square, a broad S and varying-radius loops. |
| 11–20 | Different radii and short left/right turn sequences; choose speed for each section. |
| 21–30 | Hairpins, spirals, compact serpentine routes and increasingly close branches. |
| 31–38 | An isolated sharp corner, then angular sequences and nearby branches. |
| 39–44 | Smooth continuation through one crossing, then two crossings. |
| 45–50 | Mixed compact layouts combining previously introduced manoeuvres. |

Only the first two courses are straight. All other routes have compact footprints;
the aspect ratio of their smallest enclosing rectangle is at most 1.5, independently
of orientation. Rotations and reflections alone do not make new routes. Course shapes,
turn counts, radii, branch spacing and placement of technical sections vary.
All routes remain open: return from the red endpoint along the same line.

Validation has 2, 3, 3, 2, 2 and 3 independent courses corresponding to the six
non-straight groups above. These are a finite practice curriculum, not an exhaustive
set: repeat training with different robot seeds, and use validation to compare
controllers. Robot seeds are sampled independently of the geometry generator.

Their canonical spline geometry
is in `labs/line_following/resources/courses.npz`; preparation copies it into
`data/line_following/courses.npz`. The adjacent `train.csv` and `validation.csv` list
fixed episodes as `course,seed`, where the seed chooses the robot parameters and
sensor randomness. To reproduce one of those episodes in the viewer, select its
**Course**, enter that **Robot seed**, and set **Batch** to 1. The viewer's default
seed is an exploration setting, not the seed from the benchmark CSV.

For direct Python access, use `load_courses(Path("data/line_following/courses.npz"))`
from `coursekit.labs.line_following_course`; each course provides its name, length,
and `at(s)` to evaluate the centreline at arc length `s`. The
[batched example](#running-and-writing-a-solution) loads complete episodes from CSV.

To drive the example controller on a fresh checkout, click **Create baseline** next
to the solution path. It copies the template into `work/line_following/baseline` and
runs its preparation procedure, then enables **Start**. Existing solution code is never
replaced. For an existing working copy, **Prepare solution** runs its `train.py` and
writes artifacts, just like `step_2_train`. Previewing courses performs no training.

The engine prepares in the background as soon as the viewer opens, without loading
student code. **Start** begins driving automatically; if preparation is still in
progress, it queues the run until ready. **Cancel preparation** cancels both the
preparation and any queued start. Changing **Course** during or after a run starts a
fresh run immediately. **Restart** reloads the selected solution with the current
seed and batch size. **Pause / Continue** and Space control the current run.

Use **Save recording** explicitly for runs you want to keep. Switching runs or
closing the viewer does not ask for confirmation. The viewer keeps its physics
process warm between runs: changing the course, seed or solution reuses compiled
kernels at the same batch size. A new batch size needs its own specialization.
A blocked controller or compiler is terminated on Stop or close; the next run
then starts a new process. The controller itself is loaded afresh on every run.
New viewer processes load saved kernels from a local user cache. Changing the
engine or Python/PyTorch environment invalidates that cache; see the
[compilation notes](dynamics.md) for its location and limitations.

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
area. The 94×8 mm row fits between the inner wheel edges, centred on the axle.
The fixed 11.75 mm line spans two sectors in the nominal sensor pose.

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
corners, sections that pass close to one another, and self-intersections. Close
parallel branches can be 47 mm apart (half the sensor width), so a single reading
can contain two separate lines.

At a crossing, follow the smooth continuation of the branch on which you arrived.
The crossing angle is sufficiently large to distinguish the branches, and both
branches are smooth in a neighbourhood of the crossing. Sharp corners are kept
outside that neighbourhood. The pilot requires crossing angles of at least 50°
and keeps marked corners at least 0.2 m from crossings.

The endpoint markers are coloured portions of the line itself, with the same
11.75 mm width as its dark middle section:

| Surface | Location along the route | Nominal RGB in `[0, 1]` |
| --- | --- | --- |
| Blue start / return finish | First 120 mm | `(0.08, 0.25, 0.90)` |
| Red far endpoint / turnaround | Last 120 mm | `(0.90, 0.12, 0.08)` |
| Dark line | Between the endpoint markers | `(0.06, 0.06, 0.06)` |
| Light ground | Outside the line | `(0.94, 0.94, 0.94)` |

Marker length is measured along the curve. These colours remain fixed on the
ground on both legs: return from red to blue. At the far endpoint, follow the same
route back. Driving forward or backward is allowed; rotating the body is optional.
There is no separate marker flag in the observation. The same transverse RGB sensor
sees the paint, averaged over each sector and mixed with background near edges,
with sensor noise added. Do not rely on exact RGB equality. Colours identify
endpoints, not directions at crossings or future manoeuvres. Seeing the red area
does not itself complete the outward leg: its remaining line must also be covered.

You will receive several prepared practice courses with varied geometry. Hidden
evaluation courses will differ from them and may combine permitted situations
in ways absent from the practice collection. All courses obey the same published
rules. Expect to generalise beyond memorising the practice routes.

## Completing the route

Every part of the route centreline must pass through the 94×8 mm optical row
between the wheels, in route order. There is lateral freedom within that row.
A section under a wheel or elsewhere under the chassis earns no progress.
The judge follows the continuation of the current route segment, including at
crossings and near other branches. Acceptance uses geometry, independently of
RGB noise; a visible paint edge does not suffice if its centreline lies outside
the sensing area.

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
solution. With the current sensor-area coverage rule, a probe on the fixed
train-01..04 episodes completed the two straights and the arc in 19.405, 19.699
and 38.499 seconds. The S-shaped train-04 timed out at 120 seconds after missing
route sections; its score was zero. This does not establish performance on the
complete curriculum or benchmark. Scores from older coverage rules are not
comparable. You should measure
your own completion rate and journey time, including recovery costs, and compare
performance on familiar and unfamiliar courses. Separate outward and return
times can show whether experience from the first leg helps.

Submit a self-contained solution with reproducible training, inference and all
required artifacts, an explanation of the learned component and experiments,
and an honest declaration of AI assistance. Final inference and adaptation run
on CPU without network access, within the [published pilot limits](../../docs/environment.md).

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
over rectangles in one transverse row, 94 mm wide and 8 mm deep, between the
inner wheel edges. Line width is 11.75 mm; endpoint colours and lengths are listed
under [Courses](#courses).
Route coverage uses the ground projection of this same body-fixed 94×8 mm
rectangle at the axle centre. It uses a conservative inner approximation to the
region swept by the sensor at each
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
