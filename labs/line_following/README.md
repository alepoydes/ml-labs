# Line Following

Develop and train a controller for a simulated robot that follows a marked line
to its far endpoint and returns to the start as quickly as possible. The robot
must complete the route in order without falling. Both directions, the reversal,
stops and recovery from mistakes contribute to a single journey time.

This laboratory combines feedback control, learning from interaction, adaptation
to unknown dynamics and memory. Observations are local: the controller must
estimate motion and line direction from measurements over time. The outward leg
can provide information useful on the return leg, but its time also counts.

The supplied materials include the simulator, 50 training courses, 15 validation
courses, a batched environment and a simple reference controller. Training and
assessment use the same physics and episode rules. This release uses a pilot
parameter profile; the instructor will announce assessed parameter settings and
academic grade thresholds separately.

Read this statement for the task, data and programming interface. The
[mechanics guide](dynamics.md) derives the physical model and numerical method.
The [control and learning guide](control_hints.md) introduces balancing, state
estimation, line following and reinforcement learning. The final section of this
statement explains the [graphical simulator and debugging tools](#visualisation-and-debugging).

- [Robot and observations](#robot-and-observations)
- [Rules of the episode](#rules-of-the-episode)
- [Required work](#required-work)
- [Data and course collection](#data-and-course-collection)
- [Controller and environment API](#controller-and-environment-api)
- [Development and submission workflow](#development-and-submission-workflow)
- [Visualisation and debugging](#visualisation-and-debugging)

## Robot and observations

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
parameters belong to the selected fixed experiment and are not supplied to your
controller. They stay unchanged for the complete outward and return journey.
You can adapt using the robot's observed response to your actions.

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
in the [programming interface](#controller-and-environment-api).

During evaluation, the controller is not told its true position, distance along
the route, sampled physical parameters, which branch the judge is tracking, or
whether a movement earned progress. Endpoint colours are ordinary physical
observations; they do not indicate that the preceding route was accepted.

## Rules of the episode

### Course geometry and markers

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

The supplied practice collection contains varied geometry. Hidden
evaluation courses have different geometry and may combine permitted situations
in ways absent from the practice collection. All courses obey the same published
rules. Expect to generalise beyond memorising the practice routes.

### Covering the route

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

For example, if you leave after 0.5 m and rejoin at 0.7 m, those 0.2 m
remain missing. You must return to 0.5 m or earlier and follow the missing
section. Returning to an earlier point does not erase the progress already earned.
The same rule applies independently on the return journey.

Seeing an endpoint marker is insufficient if part of that leg was skipped.
The judge does not tell the controller whether individual movements were accepted.

### Timing, failure and score

There is one uninterrupted timed episode. The first leg is not a free scouting
run. The controller may use it to learn the route and dynamics, but that time
counts toward the same total as the return leg. All searching, stopping, and
turning time counts too.

Falling fails the episode and ends simulation immediately. Ordinary contact with
the support sphere is allowed. Falling forward or sideways is not a recovery
technique. In this release, a body tilt of 0.9 radians from vertical is a fall.

For a completed journey taking simulated time $T$, the episode score is

$$
S = \min\left(100,\;200\frac{T_{\mathrm{ref}}}{T}\right).
$$

Here $T_{\mathrm{ref}}$ is a mechanical reference time for the course and sampled robot.
It uses full knowledge of the route and dynamics to estimate fast travel along
the line's centre, accounting for speed and acceleration limits, curvature,
both directions, and the turnaround. It includes no cost for discovering the
route or identifying the robot. The public `reference_time` function implements
this approximation. Its settings will be frozen for the assessed version.

The episode time limit is $T_{\max}=30\,T_{\mathrm{ref}}$, rounded up to the next
1 ms physical step. It therefore depends on both the course and its fixed robot.
The same limit covers both legs and all recovery manoeuvres. The environment
reports the effective limit as `info["time_limit"]`; tables and the graphical
viewer show it alongside elapsed time. Finishing at the deadline earns about
6.67 points.

The reference is an approximate normalisation. Any completed run taking at most
$2\,T_{\mathrm{ref}}$ earns 100 points. Legal corner cutting within the coverage rule
or useful sliding may beat the reference; the score remains capped at 100. Taking
$\tfrac{10}{3}\,T_{\mathrm{ref}}$ earns 60 points; taking $5\,T_{\mathrm{ref}}$ earns 40.

A fall or failure to complete the journey within the episode time limit earns
zero. Partial progress earns no points. The final score is the mean over all
scheduled evaluation episodes, including failed episodes. Thus faster successful
journeys can compensate for some failures. Simulator and infrastructure failures
are reported separately; an interrupted evaluation does not yield an official
aggregate. Academic grade thresholds remain to be defined before assessment.

Results also record **accepted distance**, **route completion** and **travelled
distance**. Accepted distance counts the two legs in order, from zero to twice the
one-way course length; completion is that fraction, displayed as a percentage.
Travelled distance is the horizontal path of the axle centre, accumulated at every
1 ms physical step, including searches, reverse travel, slipping and the final step
before a fall. It is not wheel odometry. Vertical motion and rotation about a fixed
axle centre add no distance. These values explain how far a zero-point attempt got;
they do not change scores. A sensor area can cover line without the axle centre
travelling the same distance, so accepted and travelled distances need not coincide.

Episode tables distinguish completion, falling and the time limit, and show the
outward/return leg at termination. A recoverable loss of the line is not itself a
terminal outcome. Interrupted checks retain available diagnostics without an
official aggregate score; controller and infrastructure errors are reported
separately from physical outcomes.

## Required work

Implement a controller and a reproducible training or fitting procedure. You may
combine conventional control with learned components, train a policy directly,
or learn a model used by a controller. Explain what is learned and measure its
effect. Explicit identification of every physical parameter is not required.

The project supplies the physical simulator, observations, episode lifecycle,
route judge and scoring. You implement `train.py` and `solution.py`, choose
settings in `config.toml`, and save the parameters or models needed for inference
in `artifacts/`. Optional shared helpers belong in `lib/`. Develop a working copy
under `work/line_following/`; the [workflow](#development-and-submission-workflow)
creates it without modifying the supplied templates.

Two templates are provided. `starter` demonstrates the minimum interface and
commands zero motor effort. `baseline` drives with the sphere on the floor. It
tracks one connected optical stripe, estimates its direction using observation
history and the gyro, and regulates speed using the encoders. Two observed line
points estimate its direction; gyro feedback damps steering. It slows for turns,
coordinates translation with rotation, and backs up along a saved line tangent
after losing the line. It settles behind a previously observed line point, then
gathers fresh forward observations while retrying slowly. At either endpoint it
can reverse travel without turning its body. Its preparation procedure saves
hand-tuned settings; it does not learn.
When a stripe becomes unexpectedly wide, it saves the incoming line estimate.
Paint extending to both sides of that saved line confirms a crossing; it then
follows the saved tangent through merged paint at low speed, without fitting
that paint's centroid. One-sided corners and parallel hairpins retain normal tracking.
It resumes optical tracking after finding a narrow stripe near the prediction;
an unsuccessful probe returns to the normal retreat procedure.
The [control guide](control_hints.md#the-reference-controller) explains the design
and its limitations. Replace or extend the preparation procedure with learning or
fitting.

Your submission must contain reproducible training, inference code and all
required inference artifacts. In the accompanying report, explain the learned
component, experimental method and results, and declare AI assistance honestly.
Measure completion rate and total journey time, including recovery. Compare
familiar and unfamiliar courses; outward and return times can help assess the
value of memory. Include failures when comparing scores.

Practice cases can be repeated exactly; extra robots can be generated explicitly
for training. Training can use simulator diagnostics to construct rewards or learning targets. During assessment,
adaptation must use the permitted observations and action history: accepted
progress, true pose and rewards are not controller inputs. Final inference and
adaptation run on CPU without network access, within the
[published execution limits](../../docs/environment.md).

## Data and course collection

### Project directory and supplied files

In these instructions, the **project directory** is the directory containing
`pyproject.toml`, `uv.lock`, `view_lab` and the numbered `step_*` scripts. Paths in
this statement are relative to that directory. Locate those files before running
a command; the directory containing this statement, `labs/line_following/`, is
not the command working directory.

```text
project-directory/
    pyproject.toml
    uv.lock
    view_lab                 view_lab.cmd on Windows
    step_0_update_resources  and step_1_* through step_5_*; .cmd equivalents on Windows
    labs/line_following/
        README.md
        dynamics.md
        control_hints.md
        resources.toml
        resources/           bundled course files and atlas
        starter/
        baseline/
    src/coursekit/           simulator and shared tools
    data/line_following/     verified working data, created during preparation
    work/line_following/     your solution variants
    runs/line_following/     local results and training histories
    dist/                    packaged solutions
```

The course collection is bundled under `labs/line_following/resources/`.
Resource preparation copies it into `data/line_following/`, verifying file sizes
and SHA-256 hashes against `resources.toml`. The same verification applies to
resources distributed by download. Treat `data/` as managed storage; save your
experiments under `work/` and `runs/`.

| File in `data/line_following/` | Contents |
| --- | --- |
| `courses.npz` | Canonical centreline geometry for all 65 courses. |
| `train.csv` | The 50 fixed training episodes, one case identifier per row. |
| `validation.csv` | The 15 fixed validation episodes. |
| `cases.npz` | Explicit robot parameters, initial states and reproducible sensor noise for all cases. |
| `diagnostic.csv` | One extra case, `diagnostic-train-04-robot-0`, sharing train-04 geometry with another robot. |
| `index.html` | An offline atlas with course drawings, lengths and fixed case identifiers. |

### Composition of the collection

Training courses are named `train-01` through `train-50`; validation courses are
named `validation-01` through `validation-15`.

| Training courses | Validation courses | Geometric characteristics |
| --- | --- | --- |
| 01–02 | — | The same one-metre straight route with two different robots. |
| 03–10 | 01–02 | Broad curves and an S bend, rounded triangular and rectangular routes, meanders and asymmetric loop-shaped sections. |
| 11–20 | 03–05 | Compact meanders, slaloms, S bends and a spiral; varying radii and successive left/right turns. |
| 21–30 | 06–08 | Hairpins, spirals, compact serpentine routes and closely spaced parallel branches. |
| 31–38 | 09–10 | Individual sharp corners, sequences of corners and nearby branches. |
| 39–44 | 11–12 | Smooth continuation through one crossing, followed by routes with two crossings. |
| 45–50 | 13–15 | Mixed compact layouts combining the manoeuvres introduced earlier. |

Only the first two training courses are straight. Every other course has a compact
footprint: the aspect ratio of its smallest enclosing rectangle is at most 1.5,
independently of orientation. Shapes, turn counts, radii, branch spacing and the
placement of difficult sections vary; rotated or reflected copies alone do not
constitute additional layouts. All courses are open routes with distinct endpoints.

Begin experiments with the early training courses, then mix earlier and later
cases to retain basic skills while adding difficulty. The validation courses have
separate geometry corresponding to the six non-straight groups. Use them to compare
controllers and report generalisation; repeatedly selecting settings on the same
validation set also fits those choices to that set. This finite collection does
not cover every permitted situation in assessment.

A **case** specifies geometry, all robot parameters, the initial physical state,
and a reproducible sensor-noise stream. Selecting the case completely defines the
experiment. Repeating or resetting it does not draw another robot. `train-01` and
`train-02` have the same geometry but different stored physical parameters.
The extra diagnostic case does not enter training/validation aggregate scores.

The physics has no stochastic forces, friction changes or random actuator commands.
Independent zero-mean Gaussian measurement noise is added once per observation:
RGB standard deviation 0.008, gyroscope 0.004 rad/s, accelerometer 0.025 m/s²,
and encoders 0.0005 rad. RGB is then clipped to `[0, 1]`. Noise is reproducible and
independent of batch order and the controller's own RNG. Randomisation of new
robots is an explicit data-preparation operation, separate from running a case.

### Loading with the supplied functions

Prepare the data as described in the [workflow](#development-and-submission-workflow),
then run Python examples from the project directory with `uv run --locked python`.
`load_courses` reads geometry; `benchmark` also reads an episode CSV and preserves
its row order:

```python
from pathlib import Path
import numpy as np

from coursekit.labs.line_following_course import load_courses
from coursekit.labs.line_following_task import benchmark

courses = load_courses(Path("data/line_following/courses.npz"))
cases = benchmark(Path("data/line_following"), "train")
case = cases[0]
course = case.course
s = np.linspace(0.0, course.length, 101)
points = course.at(s)  # Shape (101, 2), world x/y in metres.
curvature, change = course.geometry(s)
print(case.name, course.length)
```

`benchmark(..., "validation")` loads the fixed validation episodes; validation
is its default split. `Course.at(s)` accepts a scalar or an array of distances
along the route. `Course.geometry(s)` returns signed curvature in m⁻¹ and its
derivative with respect to arc length in m⁻². Values at a sharp corner must be
interpreted using the one-sided smooth segments; the tangent is discontinuous there.
At an interior join, these methods select the following segment; at the final
endpoint, they use the last segment. Distances outside `[0, course.length]` are
clamped to the corresponding endpoint.

### CSV episode format

The split files are UTF-8 CSV with the exact single-column header `case`.
Each following row names one entry in `cases.npz`; names are distinct within a
split and row order is the evaluation order. Several cases may refer to the same
course geometry. `load_cases(data_dir, split)` in `line_following_cases` loads
`train`, `validation` or `diagnostic`; omitting the split loads all stored cases.
`benchmark(data_dir, split="validation")` uses the same loader.

### NPZ experiment format

Read `cases.npz` with `numpy.load(..., allow_pickle=False)`. Format 1 stores the
following arrays, where `K` is the number of cases (66 in this release):

| Array | Shape | Type and meaning |
| --- | --- | --- |
| `version` | `()` | Integer `1`. |
| `names`, `courses` | `(K,)` each | Unicode case identifiers and references to `courses.npz` names. |
| `noise_state` | `(K,)` | Unicode JSON descriptions of NumPy PCG64 state before the first observation. |
| `noise_std` | `(K, 4)` | Float64 deviations in RGB, gyro, accelerometer, encoder order. |
| `robot_mass` | `(K, 4)` | Float64 masses: body, left wheel, right wheel, sphere (kg). |
| `robot_inertia` | `(K, 4, 3)` | Principal inertias of those bodies (kg m²). |
| `robot_offset` | `(K, 4, 3)` | Body-frame centre offsets from the axle (m). |
| `robot_radius`, `robot_friction` | `(K, 3)` each | Radii (m), Coulomb coefficients: left wheel, right wheel, sphere. |
| `robot_stiffness`, `robot_compression` | `(K, 3)` each | Contact stiffness (N/m) and compression damping (N s/m), left/right/sphere. |
| `robot_ball_drag`, `robot_gravity` | `(K,)` each | Sphere drag (N m s), gravity (m/s²). |
| `robot_stall`, `robot_speed` | `(K, 2)` each | Left/right stall torque (N m) and no-load speed (rad/s). |
| `initial_q`, `initial_p` | `(K, 8)` each | Float64 generalised coordinates and canonical momenta. |
| `initial_spin` | `(K, 3)` | Float64 sphere angular momentum in world coordinates (kg m²/s). |
| `initial_fallen` | `(K,)` | Boolean, initially false. |

The [mechanics guide](dynamics.md) defines coordinates and conjugate momenta.
Numeric values must be finite, physical masses/inertias/radii/stiffness/drive
limits positive, and noise deviations nonnegative. No pickled objects or runtime
robot-generation seeds occur in this archive. To reconstruct a noise stream, parse
its JSON, create `np.random.Generator(np.random.PCG64(0))`, and assign the parsed
state to `generator.bit_generator.state`. Draw normal arrays in the order RGB
`(16, 3)`, gyro `(3,)`, accelerometer `(3,)`, encoders `(2,)`, including the initial
observation. Draw even for a zero deviation to preserve other sensors' streams.
Per-row generators keep batch scheduling independent.

### NPZ geometry format

`courses.npz` is a compressed NumPy archive. Read it with
`numpy.load(path, allow_pickle=False)`; it contains no Python objects. Format
version 2 has exactly the following arrays. Let `N` be the number of courses,
`M` the maximum segment count, and `m = counts[i]` for course `i`.

| Array | Shape | Stored type | Meaning |
| --- | --- | --- | --- |
| `version` | `()` | `int64` | Format identifier, equal to `2`. |
| `names` | `(N,)` | NumPy Unicode strings | Unique, nonempty course names. |
| `counts` | `(N,)` | `int64` | Number of cubic segments in each course, between 1 and 2048. |
| `controls` | `(N, M, 4, 2)` | `float64` | Four ordered `(x, y)` control points per cubic segment, in metres. |
| `knots` | `(N, M + 1)` | `float64` | Parameter values at the segment boundaries. |
| `corners` | `(N, M + 1)` | Boolean | Whether a boundary is an intentional sharp corner. |

The supplied archive has `N = 65` and `M = 42`; a loader should read these sizes
from the arrays. For course `i`, retain `controls[i, :m]`, `knots[i, :m+1]` and
`corners[i, :m+1]`. Remaining entries are zero padding, not additional segments.
The retained knots start at zero and increase strictly. The first and last corner
flags are false. Control points and knots are finite. Adjacent segments meet;
position and the first two derivatives with respect to the knot parameter are
continuous at every unmarked join (C2 continuity). A marked join preserves
position but permits a discontinuity of tangent direction.

For segment `j`, let its control points be $P_0,P_1,P_2,P_3$. For a knot parameter
$u$ between $u_j$ and $u_{j+1}$, set $t=(u-u_j)/(u_{j+1}-u_j)$. Its centreline is

$$
B_j(t)=(1-t)^3P_0+3(1-t)^2tP_1+3(1-t)t^2P_2+t^3P_3,
\qquad 0\leq t\leq1.
$$

The knot parameter is neither elapsed time nor arc length. Segment length is
$\ell_j=\int_0^1\lVert B'_j(t)\rVert\,dt$. Arc distance within a segment is the
sum of preceding segment lengths plus the same integral up to $t$.
To implement evaluation by arc length independently, integrate these speeds and
invert the accumulated length within the selected segment. Tangents never vanish.
For derivatives with respect to $u$, divide the local $t$ derivative of order
$k$ by $(u_{j+1}-u_j)^k$.

The archive contains geometry only. The line width, endpoint paint, marker length
and sensor dimensions are defined by the [episode rules](#rules-of-the-episode).
The blue and red sections occupy the first and last 0.12 m of arc length. Their
positions must be derived from the curve, rather than from a fixed number of
control points. The atlas is a visual aid; its drawings are not simulation input.

The following example reads both geometry and episodes without importing
`coursekit`:

```python
import csv
from pathlib import Path
import numpy as np

root = Path("data/line_following")
with np.load(root / "courses.npz", allow_pickle=False) as archive:
    if archive["version"].item() != 2:
        raise ValueError("Expected spline course format 2")
    controls, knots, corners = (archive[key] for key in ("controls", "knots", "corners"))
    names, counts = archive["names"], archive["counts"]
    raw_courses = {}
    for i, name in enumerate(names):
        m = int(counts[i])
        raw_courses[str(name)] = {
            "controls": controls[i, :m].copy(),
            "knots": knots[i, : m + 1].copy(),
            "corners": corners[i, : m + 1].copy(),
        }
with (root / "train.csv").open(encoding="utf-8", newline="") as stream:
    episode_names = [row["case"] for row in csv.DictReader(stream)]
with np.load(root / "cases.npz", allow_pickle=False) as archive:
    if archive["version"].item() != 1:
        raise ValueError("Expected fixed-case format 1")
    i = archive["names"].tolist().index(episode_names[0])
    course_name = str(archive["courses"][i])
    robot = {
        k.removeprefix("robot_"): archive[k][i].copy()
        for k in archive.files
        if k.startswith("robot_")
    }
    initial = {
        k.removeprefix("initial_"): archive[k][i].copy()
        for k in archive.files
        if k.startswith("initial_")
    }
    noise_state, noise_std = str(archive["noise_state"][i]), archive["noise_std"][i].copy()
p0, p1, p2, p3 = raw_courses[course_name]["controls"][0]
t = 0.5
point = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3
```

This compact example assumes verified course resources. An independent loader for
other archives should also validate the dimensions and geometric conditions above.

## Controller and environment API

### Solution files and lifecycle

The shared [submission format](../../docs/submissions.md) specifies the files in a
solution. For this lab, the required entry points are:

| Entry point | Responsibility |
| --- | --- |
| `train(data_dir, output_dir, config, seed)` in `train.py` | Learn or fit from prepared data and simulation; save inference artifacts under `output_dir`. |
| `Solution(artifact_dir, config, seed)` in `solution.py` | Load saved artifacts and initialise controller state. |
| `act(observation, reset)` | Update state from the current observations and return left/right motor commands. |

Directory arguments are `pathlib.Path` objects; `config` is a dictionary read from
`config.toml`. Training may return `None` or a dictionary of finite numeric NumPy
arrays for saved learning histories. Initialise your random generators using the
supplied seed to make experiments reproducible. This seed controls your method,
not the stored robot or sensor noise; the baseline ignores it during inference.
Load models relative to `artifact_dir`, so the solution also works after packaging and extraction.

All observations are float32 PyTorch tensors on CPU with a leading batch dimension
`B`, including when `B = 1`. Return finite float32 commands of shape `(B, 2)` in
`[-1, 1]`; columns are the left and right drives. Commands are normalised motor
inputs, not target wheel speeds. Positive equal inputs drive forward; zero input
includes electrical braking in the motor model.

The Boolean tensor `reset` has shape `(B,)`. Clear the indicated rows of controller
memory before processing their inputs; other rows continue unchanged. Memory
persists across the far endpoint and throughout the return journey. A reset must
also restore the initial state of any adaptation performed within an episode.
Assessment starts a fresh controller process for each episode; do not rely on
memory of earlier assessment episodes.

`act` runs with autograd available in both local and isolated evaluation, so online
adaptation may use `backward()`. Use `torch.no_grad()` for computations that do not
need gradients. This support does not expose the simulator state to the controller.

| Observation | Shape | Meaning |
| --- | --- | --- |
| `time` | `(B,)` | Simulated seconds since the episode began. |
| `rgb` | `(B, 16, 3)` | RGB averages in `[0, 1]`, ordered from body right (sector 0) to left (sector 15). |
| `gyro` | `(B, 3)` | Body-frame angular velocity about x/y/z in rad/s. |
| `accel` | `(B, 3)` | Body-frame specific acceleration at the axle centre in m/s². |
| `encoders` | `(B, 2)` | Left/right wheel angles relative to the body, in radians. |

The body axes are x forward, y left and z up. Specific acceleration includes the
upward support response to gravity: a stationary upright robot reads approximately
`(0, 0, 9.81)`, not zero. Neither `gyro` nor `accel` is a direct orientation reading.
Encoder increments can differ from ground displacement because wheels may slip.

### Batched environment

`Environment(cases, *, compiled=True, time_limit=None, noise=True)`
constructs one independent episode per fixed `Case`. Parameters and initial
conditions are copied from the case, and its noise stream starts from the recorded state.
Courses must be at least 0.3 m long. The environment uses float64 CPU physics with
a 1 ms physical step and a 20 ms control interval (50 observations/actions per
simulated second).
The default `time_limit=None` derives a separate deadline of `30 * T_ref` for
each row. Replacing a case on reset recalculates that row's deadline. An explicit
positive `time_limit` in seconds overrides all deadlines for local diagnostics;
assessment uses the default rule.

This example advances two robots for one control interval. It uses the prepared
working solution created in the [workflow](#development-and-submission-workflow):

```python
from pathlib import Path

from coursekit.labs.line_following_env import Environment
from coursekit.labs.line_following_task import benchmark
from coursekit.training import load_solution

cases = benchmark(Path("data/line_following"), "train")[:2]
env = Environment(cases)
controller = load_solution(Path("work/line_following/my-controller"))
observation = env.observe()
action = controller.act(observation, env.first)
observation, reward, terminated, truncated, info = env.step(action)
done = terminated | truncated
if done.any():
    observation = env.reset(done)
```

`observe()` returns the current observations; `first` gives the reset mask for the
next controller call. `step` holds commands for one control interval, ending an
episode immediately if a physical step reaches a terminal condition. Its outputs
are observations, float32 reward `(B,)`, Boolean `terminated` and `truncated` tensors
`(B,)`, and a dictionary of diagnostic NumPy arrays.

`terminated` identifies completion or falling; `truncated` identifies the time
limit, which scores zero. Finished rows and their final observations remain frozen
until explicitly reset. After `env.reset(done)`, pass the updated `env.first` to
the next `act` call. `reset(mask)` repeats exactly the selected cases, including
initial observations and noise. Optional `reset(mask, cases=[...])` replaces those
rows, with one case per selected row. Other rows retain physical/controller state.

To prepare additional training robots, call `sample_case(course, seed, name=...)`
from `coursekit.labs.line_following_cases`. The seed belongs to generation, not
to `Environment`. Save resulting cases with `save_cases` and a split with
`save_split`; save their unique geometries separately with `save_courses`.
For example, `extra = sample_case(cases[0].course, 42, name="extra-42")`, followed
by `env.reset([True, False], cases=[extra])`, replaces only the first row above.
These utilities are for training; evaluated `act()` never receives case data.

`noise=False` disables measurement noise for diagnosis without changing the robot.
For individual sensors, `dataclasses.replace(case, noise_std=(...))` creates an
explicit diagnostic case. Evaluation always uses the stored noise settings.

The default training reward is newly accepted distance divided by the complete
out-and-back route length. Revisiting an already accepted portion of the current
leg earns no additional reward; the return leg has its own progress. This reward
is a learning signal, not the competition score. Training code can construct
another reward from `info`:

| `info` key | Shape | Meaning |
| --- | --- | --- |
| `q` | `(B, 8)` | True configuration: axle x/y/z, roll/pitch/yaw, left/right wheel angles. |
| `time` | `(B,)` | Elapsed simulated seconds. |
| `progress` | `(B,)` | Accepted distance across both legs, in metres. |
| `route_length` | `(B,)` | Required distance across both legs, in metres (`2 * course.length`). |
| `completion_fraction` | `(B,)` | `progress / route_length`, from 0 to 1. |
| `travelled_distance` | `(B,)` | Horizontal path of the axle centre, in metres, including recovery and slipping. |
| `leg` | `(B,)` | `0` outward, `1` return. |
| `status` | `(B,)` | `running`, `finished`, `fallen` or `timeout`. |
| `score` | `(B,)` | Episode points; zero until successful completion. |
| `reference_time` | `(B,)` | Mechanical reference time, in seconds. |
| `time_limit` | `(B,)` | Effective episode deadline, in seconds, rounded up to a physical step. |
| `outward_time` | `(B,)` | Outward completion time, or NaN until that leg is complete. |

The accepted distance `info["progress"]` is available to the **training algorithm**
after every control interval, including intervals with zero competition points.
Its increment indicates whether the movement covered a new, continuously connected
part of the route. An unchanged value means no new distance was accepted; this
also occurs during a valid return to an earlier point. Use this feedback for reward
shaping, critic targets or diagnostics during training. For example, with the
environment and controller above:

```python
import torch

accepted_before = env.info()["progress"]
observation = env.observe()
action = controller.act(observation, env.first)
next_observation, reward, terminated, truncated, info = env.step(action)
training_feedback = {
    "accepted_distance": torch.as_tensor(info["progress"], dtype=torch.float32),
    "accepted_increment": torch.as_tensor(
        info["progress"] - accepted_before, dtype=torch.float32
    ),
}
```

Store these targets with the transition in a training buffer or use them in a
training update. The deployed controller must infer successful recovery from its
sensors and memory. Neither accepted distance nor reward is an input to `act()`
in public validation, graphical runs or instructor grading. Keep training feedback
separate from the observation dictionary and from the deployed actor's inputs.

These diagnostics are never supplied to evaluated `act` calls. `env.info()` reads
the same fields without advancing simulation. Numerical failures raise an exception
instead of returning a scored episode.
Distance counters freeze for finished rows and reset independently with their
episodes. They do not add controller observations or alter competition scoring.

Normal evaluation uses the Numba engine, including a native loop over the 20
physical steps between controller calls. Its first call includes compiler startup;
subsequent processes reuse a disk cache. Batch lengths may change without new
compilation when array types and layouts are unchanged. `compiled=False` selects
the slower PyTorch autodiff reference for study and numerical comparison. Do not
put that reference inside `torch.inference_mode()`: its kinematics requires forward
differentiation. Ordinary `torch.no_grad()` is supported. The complete environment
is not a differentiable policy-training interface; see the
[mechanics guide](dynamics.md#running-and-inspecting-the-core).

### Pilot geometry and parameter settings

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
under [Course geometry and markers](#course-geometry-and-markers).
Route coverage uses the ground projection of this same body-fixed 94×8 mm
rectangle at the axle centre. It uses a conservative inner approximation to the
region swept by the sensor at each physical step, including the endpoint footprints
and a certified translation sweep, with the additional spline-error erosion.
The convex hull of two rotated footprints is not used: it could include regions
never covered during the rotation. See the public `line_following_course` module
for the calculation.

The reference profile starts at rest, stops to reverse travel at the far endpoint,
and allows a moving finish. Marked sharp corners include a stop and an in-place
change of heading; independent drive, traction, and yaw limits give an optimistic
speed profile on the remaining line. The reference grid has at most 1 mm arc-length
spacing and includes genuine polynomial joins with limits from both sides.
Exact subdivisions do not introduce artificial reference constraints. Speed passes
use cumulative acceleration bounds; time is integrated along the spline's arc
coordinate. Legal driving may beat this approximation.

`line_following_cases.sample_robot` publishes the complete pilot distribution. Component masses and
their inertias scale together by 0.85–1.15; body centre-of-mass height scales by
0.9–1.1 with consistent inertia scaling. Wheel friction lies in 0.65–0.95, support
friction in 0.45–0.75, motor stall torque in 0.065–0.095 N m, no-load speed in
38–52 rad/s, and sphere damping in 0.00015–0.00025 N m s. Geometry and contact
stiffness use the nominal mechanics profile. Independent zero-mean Gaussian
sensor noise has standard deviations 0.008 for RGB, 0.004 rad/s for gyro,
0.025 m/s² for acceleration, and 0.0005 rad for encoders. RGB is clipped to `[0, 1]`.
The episode time limit is `30 * T_ref`; a body-up tilt of 0.9 rad is a fall.

## Development and submission workflow

Install `uv` and obtain the project as described in the
[project setup instructions](../../README.md#start-here). Open a terminal in the
[project directory](#project-directory-and-supplied-files). The scripts prepare and
use its locked Python environment automatically.

```sh
./step_0_update_resources
./step_1_new_solution line_following my-controller --template baseline
./step_2_train work/line_following/my-controller
./step_3_evaluate work/line_following/my-controller
```

On Windows use the `.cmd` counterparts, for example
`.\step_2_train.cmd work\line_following\my-controller`. The wrappers resolve
relative arguments from their own directory even if invoked from elsewhere.
For Python examples and your own scripts, set the working directory explicitly
to the project directory, then use `uv run --locked python your_script.py`.

`step_0_update_resources` prepares data for the available labs. The new-solution
command copies a template and refuses to overwrite an existing variant. Replace
`baseline` with `starter` to begin from the minimal interface. Edit your copy's
`train.py`, `solution.py` and `config.toml`, then repeat training and evaluation.
Source or configuration changes require a new completed training run before
packaging. Saved artifacts and provenance are written under the working copy's
`artifacts/`; results and numeric learning histories are stored under `runs/`.

`step_3_evaluate` runs the complete fixed validation set. It reports mean score,
completion, fall and timeout rates, mean successful journey time and mean route
completion (`mean_completion_fraction`, on a 0–1 scale). It also prints a table of
individual episode outcomes, accepted/required distance, completion percentage and
travelled distance. Local
training and evaluation run directly in Python, so ordinary prints, breakpoints
and tracebacks are available. The [development guide](../../docs/development.md)
shows the same operations as Python calls.

Under `runs/line_following/<variant>/`, `evaluate.json` stores the completed
aggregate; `evaluate.cases.json` and `evaluate.cases.csv` store episode diagnostics.
Episode numbers follow `validation.csv` row order. JSON has `complete`, `cases`
and `error` fields (null on success, otherwise exception type and message).
CSV uses the metric names from the API (`case` is one-based and `leg` is
`outward` or `return`). Fractions in saved files are 0–1; tables display percentages.
Case records also contain reference and outward times; an unfinished outward leg
has JSON null / an empty CSV field. On an error, completed cases and the interrupted
episode's last valid measurements remain available, with `complete=false` and no
aggregate result. The interrupted episode has no score. A new invocation replaces
these local result files; copy them elsewhere to retain earlier experiments.
`step_5_check_submission` writes equivalent `check.*` files after extraction.
For Python evaluation, pass a `progress` callback to `evaluate`; its events containing
`metrics` carry the same episode snapshots.

Package the completed solution and check the resulting ZIP:

```sh
./step_4_pack work/line_following/my-controller
./step_5_check_submission dist/line_following-my-controller-<hash>.zip
```

Replace `<hash>` with the filename printed by packaging. The check evaluates a
fresh extraction in your local environment; it catches missing artifacts but does
not reproduce assessment isolation or resource limits. Submit the ZIP and your
report through the instructor's designated channel. Packaging does not upload
anything. Follow the [submission format](../../docs/submissions.md) and
[AI-use declaration](../../docs/ai-use.md).

### Reference measurements and experimental comparison

The current crossing-aware baseline was evaluated on all 65 published experiments
with their fixed robots and sensor noise on 2026-10-09. Each case used its own
`30 * T_ref` deadline. The previous controller was also rerun with these deadlines
to distinguish controller changes from the additional time allowance. The scores
below were recalculated from the saved times and reference values using the current
twofold scoring factor; trajectories and completion outcomes were not rerun.

| Measurement | Previous controller, `30 * T_ref` | Current controller, `30 * T_ref` |
| --- | ---: | ---: |
| Training completions | 40 / 50 | 49 / 50 |
| Validation completions | 11 / 15 | 15 / 15 |
| Training mean score, including failures | 11.16 | 13.35 |
| Validation mean score, including failures | 10.11 | 13.20 |
| Training mean accepted coverage | 87.0% | 98.8% |
| Validation mean accepted coverage | 87.5% | 100.0% |
| Falls | 0 | 0 |

No previously successful case became a failure under the same deadline. All eight
crossing-group cases (`train-39`–`train-44`, `validation-11`–`validation-12`) now
complete both legs; the previous controller timed out on all eight even with the
larger allowance. `train-03` and `train-04` retain their 53.262 s and 95.370 s
journeys. `train-08`, which approaches faint endpoint paint on a bend, completes
in 83.973 s. The remaining failure is `train-36`: a 379.713 s timeout with 39.9%
accepted coverage, also present in the previous controller.

Public deadlines range from 47.419 to 424.369 simulated seconds. Successful
journeys take approximately 12.00–25.68 times their mechanical reference, so the
factor 30 leaves time for this conservative controller without making all cases
solvable. The baseline is not a universal route solver; these finite public
measurements do not establish performance on every admissible robot or geometry.
The following measurements retain the history of earlier revisions and limits.
Their scores used the former onefold normalisation and are not comparable with the
current score values above; journey times and completion counts remain informative.

The earlier fixed-case baseline, before crossing handling was added, was evaluated
on all 65 published training and validation experiments with their stored robots
and sensor noise on 2026-10-09. That pilot used a fixed 120-second limit.

| Measurement | Training | Validation |
| --- | ---: | ---: |
| Completed journeys | 27 / 50 | 6 / 15 |
| Mean score, including failures | 3.82 | 2.83 |
| Mean accepted coverage | 79.2% | 76.4% |
| Falls | 0 | 0 |

Every case through `train-20` completed. The published `train-04` case took
95.370 s; the additional `diagnostic-train-04-robot-0` case took 80.531 s.
Both completed the outward and return legs with accepted progress restored after
recovery. Singleton runs through the graphical simulator's worker and batched
runs produce identical observations, actions and final states for these two cases.

That revision completed 33 journeys, compared with the preceding conservative
pilot's 38 under the same time limit; its remaining 32 episodes timed out.
Difficult branches, crossings and recovery limited that revision. Small steering
fluctuations from the discretised optical measurements also remain: disabling random sensor noise does not make the
16-sector observation continuous. These measurements do not establish stability
for every admissible robot or route.

The following tables retain historical measurements of four earlier controllers
from 2026-10-08 and 2026-10-09. Their physical experiments were preserved by the
fixed-case migration. All used sensor noise, the 120-second limit and the same
mechanical, sensor-area coverage and scoring rules. Mean scores include failures
as zero. These results do not describe the current controller.

| Measurement | Earlier centroid/PD | Encoder replay (2026-10-08) | Earlier two-point recovery (2026-10-09) | Previous conservative recovery |
| --- | ---: | ---: | ---: | ---: |
| Training completions | 6 / 50 | 33 / 50 | 37 / 50 | 31 / 50 |
| Validation completions | 1 / 15 | 8 / 15 | 9 / 15 | 7 / 15 |
| Training mean score | 1.03 | 7.86 | 8.46 | 4.65 |
| Validation mean score | 0.52 | 6.02 | 7.00 | 3.31 |
| Training mean accepted coverage | 22.1% | 84.3% | 83.4% | 87.3% |
| Validation mean accepted coverage | 20.6% | 74.8% | 79.4% | 79.8% |

The previous conservative controller completed every episode from `train-01`
through `train-25`.
All 27 failures were timeouts; there were no falls. In 18 timed-out episodes,
accepted distance still increased by more than 10 mm during the final 10 seconds;
the slower driving and recovery used up the time budget. The remaining nine made
little or no accepted progress over that interval. Close branches and crossings
can still cause incorrect continuation or failed recovery.

The previous two-point controller completed every published episode through
`train-27`, with 19 timeouts and no falls overall. However, its result on `train-04`
depended on the robot: the published seed finished, while the viewer's default
seed did not. The earlier encoder-replay controller completed every episode through
`train-18`, also without falls. That revision improved repeatability on
varied robots at the cost of speed, scores and some completions under the fixed
limit. Accepted coverage remains diagnostic and does not contribute points.

Selected out-and-back times, in simulated seconds:

| Episode | Earlier centroid/PD | Encoder replay | Earlier two-point recovery | Conservative recovery |
| --- | ---: | ---: | ---: | ---: |
| `train-01` | 19.405 | 10.485 | 14.277 | 19.339 |
| `train-02` | 19.699 | 11.044 | 14.361 | 20.108 |
| `train-03` | 38.499 | 28.047 | 31.039 | 53.796 |
| `train-04` | Timeout at 120 s | 58.747 | 50.193 | 75.524 |
| `train-19` | — | Timeout at 120 s | 63.046 | 88.459 |
| `train-25` | — | Timeout at 120 s | 80.706 | 114.379 |
| `train-34` | — | Timeout at 120 s | 98.531 | Timeout at 120 s |

An additional pilot of that previous controller used `train-04` robots generated
with seeds 0–31 and the former published CSV seed.
All 33 journeys completed both legs in 68.545–87.303 simulated seconds; seed 0 took
69.686 seconds. On the gently curving section after the first bend (outward route
distance 1.1–1.7 m), maximum lateral and heading errors were 3.95 mm and 0.059 rad.
The largest yaw-rate standard deviation over that section was 0.144 rad/s.
This check detects sustained steering oscillation even when an episode finishes.

Current regression tests cover sharp and C2-rounded 90-degree turns, both left
and right, with eight explicitly generated robots: all 32 journeys complete both
legs within the public episode limit. They also check physical reverse recovery and retain bounds
of 6 mm lateral error and 0.15 rad heading error on the straight after the turn.
The eight crossing-group cases and their mirrored versions complete all 16
out-and-back journeys. Separate checks retain completion on close parallel
branches and on the `train-08` bend approaching faint endpoint paint.
An additional `train-04` check covers nine robots with normal sensor noise,
and ten ablations using the published and diagnostic robots with noise disabled
or enabled for only one sensor type. It checks both legs, recovered progress,
accumulated body rotation and time spent without progress, as well as lateral
and heading errors. The original nine noisy cases retain a 0.20 rad/s yaw-rate
standard-deviation bound on the gently curving section. In the noise-free published
case that statistic is 0.213 rad/s; its spatial and whole-journey checks still pass.

Tuning and regression checks used synthetic corners and public cases; settings
were held fixed throughout each reported sweep. These are development results,
not an unbiased estimate on unseen courses. The controller does not guarantee
completion on the assessment benchmark. Results
obtained under earlier coverage rules are not directly comparable. Assessment
parameter validation and academic grade thresholds remain separate work.

Record case identifiers/fingerprints, training seed, configuration and computational
cost of your experiments. Compare learned changes against a reference on the same
episodes, report failures as well as successful times, and examine outward and
return performance separately. A faster controller on familiar routes may still
have a lower average score on unfamiliar routes.

## Visualisation and debugging

### Offline course atlas

The [course atlas](resources/index.html) is an HTML page containing drawings of all
65 routes plus the diagnostic example, split filters, lengths and fixed case
identifiers. Open it in a web browser; no Python process or network connection is required. After resource
preparation, the same page is available at `data/line_following/index.html`.

### Graphical simulator

The graphical simulator is the desktop application for previewing courses, running
a controller and examining its behaviour. From the
[project directory](#project-directory-and-supplied-files), launch it with:

```sh
./view_lab line_following
```

On Windows use `.\view_lab.cmd line_following`. The launcher installs the locked
optional Qt and ModernGL dependencies. A desktop display and OpenGL 3.3 or newer
are required; training and command-line evaluation do not require graphics.

The application prepares the bundled courses automatically. Previewing a course
requires neither a solution nor training. Select **Course**, or use its
previous/next arrows; enable **Top view** and click **Fit track**. The status line
shows the course name and one-way length. Blue marks the start/return finish;
red marks the far endpoint. Scroll to zoom and drag to pan. **Follow** keeps
the camera centred on the robot. **Sensor area** overlays the 16 measured ground
sectors through the chassis, including the exact 94×8 mm route-coverage region.
In the large view, the RGB strip in the scene's upper-left corner displays readings
in body order, from right (0) to left (15). It is hidden in the compact preview.

To reproduce a fixed episode, select its case identifier under **Course** and set
**Batch** to 1. Larger batches use consecutive fixed cases; **View robot** chooses
the displayed row. Open **Setup…** to turn **Sensor noise** off. Such a run is labelled
as diagnostic and recorded with that setting. Normal evaluation keeps noise enabled.

The **Next run** selector distinguishes **Built-in baseline** from **Working
solution**. The first launch selects the built-in baseline. **Start** prepares a
managed copy of the installed template under `runs/line_following/builtin/`, then
starts the episode. It does not select an older copy merely because its directory
is named `baseline`. Previewing courses performs no training. The controller choice,
working path and course are retained in `runs/line_following/viewer.json` between
openings.

Open **Setup…** for **Prepare solution**, **New copy from baseline**, the sensor noise
setting and course preparation. **New copy from baseline** creates, prepares and
selects a numbered working copy under `work/line_following/baseline-current[-N]`.
Use it to start again from the installed example while preserving earlier code and
parameters. For a working copy, **Prepare solution** runs that copy's `train.py`,
just like `step_2_train`; it does not replace the algorithm with the baseline.
The preparation status checks source and configuration fingerprints. Edited code
remains runnable for local debugging
with a notice that preparation is no longer current. Saved parameter hashes are
checked at loading; changed or historically unverified parameters are identified
in the loaded controller's status and details. Prepare again to record a matching
set. To select your own prepared solution at launch:

```sh
./view_lab line_following --solution work/line_following/my-controller
```

**Running controller** identifies what the process actually loaded, independently
of **Next run**. **Details…** shows its path and fingerprints of code,
configuration and saved parameters, together with preparation status. These values
are captured at loading and retained in recordings; choosing another solution or
editing files does not change the identity of the existing run. When the run ends,
the label becomes **Last run controller**. Playback shows **Recorded controller**;
older recordings without this information explicitly show an unknown identity.

The engine prepares in the background when the application opens, without loading
controller code. **Start** begins motion automatically; during preparation it queues
the run. **Cancel preparation** cancels preparation and any queued start. Changing
**Course** during or after a run starts a fresh run. **Restart** reloads the selected
solution with the selected cases and batch size, including a fresh controller state.
Restart clears the controller's memory; it does not train or reset a working copy's
saved parameters. Use **Prepare solution** to prepare that copy again, or **New copy
from baseline** for a fresh reference controller.
The next solution, course and batch remain visible during simulation. The optional
setup controls can be expanded when needed.

### Time controls, diagnostics and recordings

The **Window** menu offers minimize, maximize and fullscreen. F11 toggles fullscreen;
the viewer restores its size and window mode on the next launch.
The viewer opens with a large 3D scene below the controls. Double-click the scene
to open diagnostics. It becomes a live preview filling the upper band to the right
of the solution, course and run controls; plots fill the area below. A single plot
grows to use the available height. The status of the selected robot appears beside
**Fit track** when the window is wide enough, and below the camera controls when
it is narrow.
Double-click the preview to return to the large scene. A tooltip on the scene
explains this gesture.

**Pause / Continue** and Space control the active run. While paused, **Step 1 ms**
advances one physical step and **Step control** advances one controller interval.
The timeline allows inspection of recorded history. Continuing returns to the latest
live state; seeking does not rerun the controller or branch simulation from an old
state. Playback offers a speed selector, previous/next frame and **Next control**
navigation between recorded calls. Its timeline sits to the right of **Speed**.
Space controls playback when no run is active.

An optional `debug(index=0)` method returns small diagnostic values for a batch row.
The application copies them after each action. The diagnostic table displays
only fields explicitly returned by this method, including any sensor values.
The starter and baseline inherit from `coursekit.controller.Controller`:
`super().__init__()` initialises the sensor snapshot and
`super().act(observation, reset)` copies the current observations. The base
`debug(index)` exports the sensors for that row, with RGB split into three
16-element vectors. Existing controllers need not inherit from this class.

Extend the default dictionary in your own controller, for example:

```python
def debug(self, index=0):
    return super().debug(index) | {
        "line_offset_m": float(self.error[index]),
        "target_speed_m_s": float(self.target_speed[index]),
        "line_weights": self.line_weights[index],
        "motor_commands": self.command[index],
    }
```

The baseline additionally demonstrates visibility flags, recovery modes,
estimated speed, measured/target yaw rate and the steering error actually used.
These are controller estimates, not true line geometry or simulator pose.

Check fields in **Diagnostics** to compare them on a plot; checking a field also
selects it for the current-value inspector. Expand vectors to see and choose
components. The field drawer slides out from the left edge of the plots and returns
behind that edge; use its tab or **Panels → Diagnostic fields**. Each curve has a
stable colour and legend entry. Click a plot panel to choose which plot the
checkboxes edit. **Add plot** below the plots creates another; each plot has its own
remove button. The **Panels** menu can hide or reveal a plot without deleting it.
**Heatmap** displays a vector's history; **X–Y** compares two
scalar channels, including components of the same vector. **Value** displays a
current matrix. A useful baseline layout combines speed/target speed, yaw/target
yaw and a heatmap of `line_weights`.

Click a plot to pause and seek the nearest recorded controller call. All panels
and the robot view follow that position. **Follow time** scrolls a selected time
window, **Fit all** fits the history, and pan/zoom preserves a manually chosen
interval. Plot selections, the active plot, colours, modes and panel visibility persist
between launches. The `debug()` method must not change controller behaviour. See the
[diagnostic value contract](../../docs/development.md#diagnostics-and-recordings)
for supported values and limits. The judge never calls this method.

Use **Save…** explicitly before replacing or closing a run you want to
keep. Routine switching and closing do not ask for confirmation. Saving during a
run stores received frames so far; later frames require another save. A recording
contains courses, sampled physical parameters, poses, observations, actions and
diagnostics. It supports inspection without executing solution code; it is not a
resumable simulator checkpoint or an inference artifact.
The status display shows accepted distance, completion percentage, travelled
distance and the episode outcome for the selected robot and frame. New recordings
save travelled distance directly. Compatible older recordings without this field
show a dash; the viewer does not infer an exact path length from sparse frames.

Robot recordings use format version 4 and include case identifiers/fingerprints,
noise configuration, case-format version and sensor dimensions, line width,
marker length, paint colours and the coverage rule version. Both the local and
instructor applications reject older or incompatible recordings without replacing
the currently open view. Use the matching coursekit environment to inspect such
historical recordings, or create a new recording. Course archives retain their
separate format version 2; controller observations and actions retain their API.

### Compilation and stopping a run

The physics process stays available between runs. Changing cases,
solutions or batch lengths reuses the Numba functions with matching array types
and layouts. New application processes load saved functions from a local user
cache; engine or Python/Numba changes invalidate that cache. Compilation is
separate from simulated time. See the [compilation notes](dynamics.md#running-and-inspecting-the-core)
for cache location, first-run cost and limitations.

**Stop** ends a run and retains received frames. A blocked controller or compiler
is terminated on Stop or close; the next run starts a new process. The controller
is loaded afresh on every run even when the physics process is reused. The shared
[graphical tools guide](../../docs/development.md#graphical-inspection-and-manual-play)
also describes window controls, diagnostic plots and playback shortcuts.
