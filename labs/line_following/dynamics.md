# Robot dynamics

The dynamics prototype is a single [public PyTorch module](../../src/coursekit/labs/line_following.py).
Instructor experiments import this same module. It needs no additional Python
dependencies. This is the mechanical core, not yet a complete lab environment:
tracks, observations, route coverage, parameter sampling, and scoring are pending.
The nominal robot and numerical settings are illustrative, not assessment limits.

## Coordinates and energy

Use SI units and float64 tensors. The world has a horizontal floor at $z=0$.
The body frame has $x$ forward, $y$ left, and $z$ up. The configuration is

$$
q=(x,y,z,\phi,\theta,\psi,\alpha_L,\alpha_R).
$$

The first three coordinates locate the axle midpoint. Body rotation is
$R=R_z(\psi)R_y(\theta)R_x(\phi)$. Wheel angles are relative to the body.
Positive wheel rotation drives forward; positive pitch leans forward. The sphere
centre is fixed behind the axle. Its orientation is irrelevant because it is
homogeneous and isotropic; its world angular momentum is a separate state.

Each centre has position $c_i=x+Rr_i$. `torch.func.jvp` differentiates this
pose map to obtain translational velocities and $\dot R$; angular velocity is
read from $\dot R R^T$. Add wheel spin about the axle to obtain each wheel's
angular velocity. Translational and rotational kinetic energies give

$$
L(q,v)=T(q,v)-V(q),\qquad
p=\partial_v L,\qquad M(q)=\partial_v^2 L,\qquad v=M(q)^{-1}p.
$$

`grad` and `jacrev` compute these derivatives. Linear systems are solved without
forming an inverse. Sphere rotational energy is accounted for separately as
$\lVert s\rVert^2/(2I_b)$; its coupling to the body comes from bearing damping
and floor contact, not a rigid rotational attachment.

## Contacts, motors, and losses

Wheel contacts use thin circular tyres; the sphere uses its lowest point. For a
contact's signed height $g$, normal elasticity contributes
$k\max(-g,0)^2/2$ to the potential. Thus elastic load is $k\max(-g,0)$.
Compression adds viscous damping; separation never produces an attractive force.
This is a compliant contact model with small, measurable penetration. It is not
a rigid impact or detailed tyre deformation model.

The contact velocity is the velocity of the material touching the floor:
$v_c+\omega\times r_c$. Differentiating only the geometric lowest point would
incorrectly remove wheel spin from slip. Contact Jacobians map world impulses
to generalized momenta, so changing tilt and load affects both traction and body
motion. No rolling constraint forces wheel rotation to equal ground travel.

At each contact, longitudinal and lateral friction share a single disk
$\lVert\lambda_t\rVert\leq\mu\lambda_n$. Projected iterations minimize kinetic
energy at fixed configuration inside these disks. They approximate sticking and
sliding without selecting a preferred sliding direction. There are 16 iterations
per resistance substep; their convergence is tested. Normal damping is solved
before tangential friction. A separated contact has neither damping nor friction.

Sphere bearing damping acts on world relative angular velocity
$\omega_b-\omega_{body}$, with equal and opposite torques. An implicit solve
dissipates its energy even when the sphere is airborne. Lifting the sphere
removes its floor friction, not its inertia or internal bearing resistance.

Motor actions are normalized to $[-1,1]$ and clipped to that interval. The motor
law is

$$
\tau_i=\tau_{stall,i}\,\operatorname{clip}
\left(a_i-\dot\alpha_i/\omega_{0,i},-1,1\right).
$$

This represents a current-limited drive with back EMF. Zero action applies
electrical braking. Torques act on relative wheel coordinates; the body's
reaction follows from the coupled inertia. Motor impulses use an explicit
substep, so time-step convergence and passive energy are checked at the nominal
settings. Arbitrarily large steps or arbitrary replacement parameters are not
validated by those checks.

## Integration and failure

The conservative core uses the midpoint discrete Lagrangian

$$
L_d(q_0,q_1)=hL\left((q_0+q_1)/2,(q_1-q_0)/h\right),\qquad
p_0=-D_1L_d,\quad p_1=D_2L_d.
$$

Five iterations solve the first equation, reusing a factorized Jacobian.
The remaining residual must pass `check`. This $(q,p)$ map is second order and
symplectic where the potential is smooth, up to the nonlinear solve error. It
also includes gravity and contact elasticity. Symplecticity does not mean exact
energy conservation. Contact activation is nonsmooth.

A half resistance step precedes and follows the conservative step. Resistance
contains first-order dissipative solves, so the complete method is not generally
second order or symplectic. It should dissipate energy when unpowered, and its
trajectories must converge as the step decreases.

The nominal physical step is 1 ms. A body tilt of 0.9 radians from vertical marks
a fall; later calls freeze that terminal state. Euler angles stay in a local
chart before falling. Tumbling and collisions of the fallen body are omitted.
Numerical failure is distinct: non-finite values or excessive solve residuals
raise `FloatingPointError` through `check` and must never be recorded as a fall
or a score. Diagnostic impulses contain normal damping and tangential friction;
elastic normal forces are already integrated by the conservative core.

## Running and inspecting the core

From the student project environment:

```python
import torch
from coursekit.labs.line_following import check, initial_state, robot, step

torch.set_num_threads(1)
model = robot()
state = initial_state(model)
action = state.q.new_tensor([0.2, 0.2])
advance = torch.compile(step, fullgraph=True)
for _ in range(1000):
    state, diagnostics = advance(state, action, model)
    check(state, diagnostics)
```

Use `advance = step` to inspect ordinary Python execution. Compilation needs the
PyTorch-supported compiler toolchain and has a substantial first-call cost;
steady simulation reuses the compiled graph. It does not select different physics.
`torch.vmap(step, in_dims=(0, 0, None))` runs stacked states and actions with one
shared robot; use `in_dims=(0, 0, 0)` for stacked parameter sets. Compile the
batched function for training. Do not put these calculations in
`torch.inference_mode`, which disables the forward differentiation used for
kinematics. A complete rollout is not a smooth function at impacts, friction
transitions, or termination; autodiff mechanics alone does not promise useful
policy gradients through these events.

`initial_state` places equal-radius wheels and the sphere at first contact,
without precompressing their springs. A brief settling transient is expected.
The initial-state helper assumes the nominal axle and rear-sphere geometry.
`Robot` stores explicit parameters so that experiments can change them; an
assessment parameter family still requires calibration and validation.

The [physics tests](../../tests/test_line_following.py) cover independent
kinematic checks, positive inertia, free fall, conserved momenta, symplecticity,
time reversal, passive losses, friction disks, solver and time-step convergence,
terminal falls, batching, and compiler graph capture.
