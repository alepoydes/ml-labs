# Control and learning: a starting point

This guide introduces ideas that can help you design experiments for
[Line Following](README.md). It assumes basic linear algebra, derivatives and
familiarity with PyTorch. The [mechanics guide](dynamics.md) defines the actual
robot and simulator. The reduced models below explain control principles; their
approximations need to be checked against that simulator.

A useful sequence of experiments is supported driving, balancing near the upright
position, slow line following, and then speed selection and adaptation. These are
possible components of a solution, not required implementation stages. You may
learn a complete policy instead. In either case, distinguish the true physical
state, the measurements available to the controller, and its internal estimates.

## Why moving the wheels can prevent a fall

Consider a point mass on an upright rigid rod of length $\ell$, attached to a
horizontally moving support. Let $\theta$ be the forward lean angle and $p$ the
support position. Near the upright position, its equation is approximately

$$
\ddot\theta = \frac{g}{\ell}\theta-\frac{1}{\ell}\ddot p.
$$

With a stationary support, a forward lean produces further forward angular
acceleration: the equilibrium is unstable. Accelerating the support towards the
fall can counteract that acceleration. For example, an ideal support-acceleration
command $\ddot p=k_p\theta+k_d\dot\theta$ gives

$$
\ddot\theta+\frac{k_d}{\ell}\dot\theta
+\frac{k_p-g}{\ell}\theta=0.
$$

This simplified system is stable when $k_p>g$ and $k_d>0$. The first term in the
command corrects the lean; the second damps angular motion. These gains command
an ideal acceleration, whereas the lab robot accepts bounded motor inputs. Wheel
inertia, motor reaction torque, slip and contact loads change the relationship
between an input and acceleration. The calculation explains the mechanism without
providing gains for the simulated robot.

Balancing angle alone also permits the robot to drift. A practical controller
usually relates balance to longitudinal velocity or position error. One possible
architecture has a slower velocity loop that requests a small lean, a faster
balance loop that controls the common motor input, and a steering loop that
controls the difference between the two motors. Their requests share the same
actuation and traction limits.

```mermaid
flowchart LR
    sensors[RGB, IMU and encoders] --> estimates[State and line estimates]
    estimates --> balance[Balance and speed control]
    estimates --> steering[Steering]
    balance --> motors[Motor command allocation]
    steering --> motors
    motors --> robot[Robot and contacts]
    robot --> sensors
```

The rear sphere changes the mechanics. When it supports the body, its contact
constrains lean and adds resistance. A controller derived near upright balance
need not work while resting on the sphere or during lift-off. Examine these regimes
separately before relying on a single local model.

## A minimal linear control model

### State, input and linearisation

Write nonlinear dynamics as $\dot x=f(x,u)$. Around an equilibrium
$f(x_*,u_*)=0$, define deviations $\delta x=x-x_*$ and $\delta u=u-u_*$. A first-order
approximation is

$$
\delta\dot x=A\delta x+B\delta u,\qquad
A=\left.\frac{\partial f}{\partial x}\right|_*,\quad
B=\left.\frac{\partial f}{\partial u}\right|_*.
$$

A reduced balancing state might contain displacement, velocity, pitch and pitch
rate. Such a model describes a neighbourhood of an operating condition. It does
not include every contact or friction transition in the full simulator. You can
estimate a local response from small input perturbations or derive a reduced
mechanical model. Compare predicted and measured responses before designing
feedback from it.

Under state feedback $\delta u=-K\delta x$, the linearised dynamics become
$\delta\dot x=(A-BK)\delta x$. In continuous time, asymptotic stability requires
all eigenvalues of $A-BK$ to have negative real parts. Their imaginary parts and
magnitudes affect oscillation and decay rates. Large gains can demand more torque
than the motors supply or amplify measurement noise.

The controller acts every $\Delta t=0.02$ s. For a linear model with input held
constant between calls, the corresponding sampled model is

$$
\delta x_{k+1}=A_d\delta x_k+B_d\delta u_k,\qquad
A_d=e^{A\Delta t},\quad B_d=\int_0^{\Delta t}e^{A\tau}B\,d\tau.
$$

Its closed-loop eigenvalues must lie inside the unit circle. A controller designed
as though measurements and corrections were continuous can behave differently at
this sampling rate. The simulator's 1 ms integration step does not provide the
controller with measurements every millisecond.

### PD, PID and LQR

For a scalar error $e$, a PID law combines proportional correction, accumulated
error and its rate of change:

$$
u=k_p e+k_i\int e\,dt+k_d\dot e.
$$

A PD law omits the integral. Integral action can compensate a persistent bias, but
an integral that keeps growing while the motor is saturated causes *windup*.
Limit or suspend accumulation when appropriate. Differentiating noisy measurements
also amplifies noise; measured angular rate or a filtered difference is often more
useful. Define the error sign and check the response to a small perturbation before
increasing gains.

Linear quadratic regulation (LQR) chooses a feedback matrix for a linear model
using a quadratic cost, for example in discrete time:

$$
J=\sum_{k=0}^{\infty}
\left(\delta x_k^TQ\delta x_k+\delta u_k^TR\delta u_k\right),
\qquad Q\succeq0,\quad R\succ0.
$$

Under the usual stabilisability and detectability conditions, a Riccati equation
gives a stabilising feedback law. The weights describe the relative cost of state
errors and effort; their units and scaling matter. This is a local design tool.
It does not enforce the robot's input bounds or guarantee stability after loss of
traction. Russ Tedrake's [LQR notes](https://underactuated.mit.edu/lqr.html) explain
the derivation and local control of nonlinear systems.

## Estimating the quantities a controller needs

The controller receives RGB, body angular velocity, specific acceleration,
body-relative wheel angles and time. It receives neither pitch nor true ground
velocity. A controller written in terms of those quantities therefore needs an
estimator.

Near planar motion, the y component of `gyro` approximates pitch rate. Integrating
it estimates pitch change, but noise accumulates and the initial angle still
matters. With negligible translational acceleration and roll, an accelerometer
estimate under this lab's coordinate convention is

$$
\theta_{acc}=\operatorname{atan2}(-a_x,a_z).
$$

During acceleration, the same measurement contains both gravity's contribution
and motion. Treating it as a direct tilt measurement can produce an incorrect
correction precisely when the robot is accelerating to balance.

A complementary filter illustrates how two imperfect signals can be combined:

$$
\hat\theta_k=\beta\left(\hat\theta_{k-1}+\Delta t\,\omega_{y,k}\right)
+(1-\beta)\theta_{acc,k},\qquad 0<\beta<1.
$$

This is a planar approximation. Choose the time scale deliberately, initialise it
on episode reset, and inspect its error during acceleration and contact changes.
For larger rotations, body gyro components are not individually equal to Euler
angle rates; use the rotation kinematics from the mechanics guide.

Encoder differences estimate relative wheel rotation. Converting them to ground
velocity assumes rolling and must account for body rotation as well as wheel
rotation. Slip violates that assumption. Gyro integration can provide short-term
heading changes, but there is no absolute compass measurement. A model-based
observer predicts motion and corrects it from sensor residuals; an extended Kalman
filter is one possible extension when the nonlinear model and noise assumptions
are adequate. The [output-feedback notes](https://underactuated.mit.edu/output_feedback.html)
discuss state estimation together with control.

During training you can compare estimates with the true state in `info`. Keep that
comparison in experiment code: the evaluated controller must update its estimates
using only its observations and actions.

## Following a line with conventional control

### Extracting an error from the optical row

For nominal sensor width $W=0.094$ m, the transverse centre of sector $i$ is

$$
y_i=W\left(\frac{i+1/2}{16}-\frac12\right),\qquad i=0,\ldots,15.
$$

Positive $y$ is to the robot's left. Let $w_i\geq0$ measure evidence of line paint
in that sector, derived from brightness or colour. When enough paint is visible,
a weighted centroid gives a simple lateral observation:

$$
e=\frac{\sum_iw_i y_i}{\sum_iw_i}.
$$

Retain a separate visibility/confidence measure. An empty row must not be
interpreted as a perfectly centred line. Blue and red markers are also line
paint; a detector designed only for neutral dark pixels may lose them. Sector
averaging, noise and tilt affect the measurements, so sub-sector offsets are
estimates rather than exact geometric coordinates.

A single transverse observation does not determine line heading. Changes in
lateral error, together with angular velocity and estimated forward motion,
provide evidence over time. The same error can evolve differently during forward
and reverse travel. This is a reason to retain a short measurement history or an
explicit heading estimate.

### Steering, speed and recovery

For common input $u$ and differential input $d$, commands can be expressed as

$$
a_L=u-d,\qquad a_R=u+d.
$$

In nominal forward rolling, positive $d$ favours a left turn. A basic line follower
can use lateral error and its change to request steering, with gyro feedback to
damp yaw. Check the signs during reverse travel. Clipping the two final commands
independently can alter both the requested common and differential components;
available motor effort has to serve balance, travel and steering together.

At higher speed, the robot covers more distance before its next observation and
has less time to correct an error. Reducing speed when line confidence decreases,
steering demand rises or recent errors grow is one possible conventional strategy.
Since the sensor does not look ahead, an unseen sharp turn can still require a
stop and recovery.

When the line disappears, its last observed side and recent motion can guide a
short search or a return to an earlier location. Remember that finding paint again
does not prove that the missed route interval was covered. The judge does not
report that event to the controller. At close branches, a centroid of two lines
can fall on unpainted ground between them; distinct candidate lines and temporal
continuity become useful. At crossings, the intended continuation is smooth.
Endpoint colour detection also needs memory: entering red paint is earlier than
covering its final point, and it does not prove that the outward leg was accepted.

These limitations explain the role of the supplied baseline. It gives a small,
inspectable feedback controller and a reproducible reference, while leaving
robust perception, recovery, balance and speed selection to your solution.

## Learning a controller from interaction

### Policy, reward and memory

In reinforcement learning, an agent chooses actions and learns from the resulting
rewards. Here a policy receives observation $o_t$ and may maintain memory $m_t$:

$$
m_{t+1}=F_\vartheta(m_t,o_t),\qquad
a_t\sim\pi_\vartheta(\,\cdot\mid o_t,m_t).
$$

Training seeks a high expected return, commonly
$\mathbb E[\sum_{t=0}^{T-1}\gamma^t r_t]$, with discount $0<\gamma\leq1$ and $T$
control transitions in the episode. Memory is useful because the current sensor
row does not reveal the full state, route direction or robot parameters. It can
be an explicit estimator, a short observation history or a recurrent network.
Sutton and Barto's [Reinforcement Learning: An Introduction](https://mitpress.mit.edu/9780262039246/reinforcement-learning/)
provides the background on returns, value functions and policy learning.

The default progress reward helps discover motion along the route, but by itself
it does not reproduce the time-based competition score. Partial progress can give
positive training reward even when the episode ultimately fails. Time cost,
success/failure outcomes and control regularisation are possible additions; check
whether they improve the actual average score. A poorly chosen penalty can make
remaining still preferable to attempting the course. A reward for repeated visible
paint can encourage behaviour that earns no route progress.

### Approaches to investigate

Learning can begin with a small number of controller or estimator parameters.
Optimise them on varied training episodes and compare with their initial settings.
Another option is to imitate a conventional controller, then improve the policy
through interaction. A learned residual can adjust the output of an existing
controller; this reduces the initial search space but does not automatically
preserve its stability.

For direct continuous-action learning, actor–critic methods combine an action
policy with an estimate of future return. The critic helps evaluate actions;
a fixed policy can act using the actor and its state alone. Two examples are
[Proximal Policy Optimization](https://arxiv.org/abs/1707.06347), which discourages
large policy changes through a clipped optimisation objective, and
[Soft Actor-Critic](https://arxiv.org/abs/1801.01290), which reuses stored transitions
and includes an entropy objective. Neither method is prescribed. Understanding
observation history, rewards and episode boundaries matters before adapting an
algorithm to this task.

The simulator can supply training targets or a critic with information unavailable
at deployment. The final actor, including any online update, must still operate
from the documented observation stream. Learning a dynamics model for short-horizon
planning is another possibility; model errors near slip and changing contacts
should be measured explicitly.

### Organising robot experiments

Use the batched environment to collect experience from several episodes. Keep its
batch size fixed to reuse compilation; reset completed rows independently and
clear only their controller memories. Preserve the final observation before
resetting. In this lab, the 120 s timeout is a scored failure: although the API
labels it `truncated`, it is not merely an arbitrary training rollout cutoff.
Decide value targets accordingly. A rollout buffer boundary while an episode is
still running is a different event.

Vary courses and robot seeds during training, retaining simpler cases as difficulty
increases. Evaluate with fixed seeds for reproducible comparisons and with additional
seeds to estimate sensitivity to dynamics. Keep model selection separate from the
experiments used to report generalisation. Record training cost and failed episodes,
and compare against the conventional controller under the same conditions.

Finally, automatic differentiation inside the mechanics does not make
`Environment.step` differentiable through an entire trajectory. The environment
detaches actions and uses NumPy for geometry and judging; contacts and termination
also introduce nonsmooth events. Policy-gradient methods learn from collected
transitions and need no derivative through that simulator. Gradients computed
inside `act` for a learned model or online adaptation are a separate capability.
