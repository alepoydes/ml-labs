# Line Following

Train a controller for a simulated robot that follows a line to a marked endpoint,
turns around, and returns to the start as quickly as possible. The whole journey
is timed, including both directions, the turnaround, stops, and recovery from
mistakes. The robot must remain upright and cover every part of the route in
order.

This is the design description for the second laboratory. A public dynamics
prototype is available in the accompanying [mechanics guide](dynamics.md).
Tracks, sensors, training resources, and evaluation are not yet implemented.
Final numerical settings, the controller interface, and the grading formula
will be supplied before the assessed release.

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

The physical model and allowed parameter ranges will be public. The actual
parameters are chosen at the start of each episode and are not supplied to your
controller. They stay unchanged for the complete outward and return journey.
You can adapt using the robot's observed response to your actions.

## Observations and control

Your controller will receive one transverse row of 16 colour measurements from
under the chassis, together with gyroscope, accelerometer, and wheel-encoder
readings. Each optical sector reports the average RGB colour over its sensing
area. The line has a fixed physical width, intended to span approximately two
sectors in the nominal sensor pose.

The row sees only its designated area beneath the chassis. It does not provide
a forward view or a map. You can use the history of observations and actions to
estimate the local direction of the line, motion, and dynamics.

The controller commands the left and right drives. Its memory and internal
estimates persist through the turnaround and return journey, and are reset
between independent episodes. Exact sensor and action units, rates, and limits
will be specified with the simulator.

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
outside that neighbourhood. Precise geometric bounds will be published before
assessment.

Distinct coloured sections of the line mark the start and the far endpoint.
At the far endpoint, turn the robot around and follow the same route back. The
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
technique. The exact fall boundary and episode time limit will be published with
the physical settings.

The objective is a reliably completed journey with a short total time. Evaluation
will record completion, elapsed time, accepted progress, and failure outcomes.
An incomplete journey is not a successful finish. The numerical treatment of
failures, aggregation across courses, and academic grade thresholds remain to be
defined before assessment.

## Training and reporting

You may repeat practice episodes with different permitted physical parameters,
use their outcomes for learning, and compare different controllers. Training
rewards can guide optimisation, but accepted progress and rewards are not part
of the acting controller's evaluation observations. Within an episode, adaptation
must use its permitted observations and action history.

A simple conservative reference controller is planned to provide a starting
comparison. Its performance has not yet been measured. You should measure your
own completion rate and journey time, including recovery costs, and compare
performance on familiar and unfamiliar courses. Separate outward and return
times can show whether experience from the first leg helps.

Submit a self-contained solution with reproducible training, inference and all
required artifacts, an explanation of the learned component and experiments,
and an honest declaration of AI assistance. Final inference and adaptation run
on CPU without network access, within limits to be published before release.
