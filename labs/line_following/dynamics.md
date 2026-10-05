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

The implementation avoids the velocity Hessian. Stack linear and angular
velocities weighted by square roots of masses and inertias into $w(q,v)$, so that
$T=\lVert w\rVert^2/2$. At fixed $q$, this map is linear in $v$: evaluating it
on the eight basis vectors gives the columns of $J$, and $M=J^T J$ exactly.
This uses the same kinetic-energy definition, not a separately maintained mass
formula or finite differences. `grad` computes momenta and the first derivatives
of the discrete Lagrangian. Sphere rotational energy is accounted for separately as
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

Five Broyden iterations solve the first equation without its coordinate Hessian.
For $F(q_1)=p_0+D_1L_d(q_0,q_1)$, the initial inverse Jacobian approximation is
$H_0=-hM(q_0)^{-1}$, obtained with one linear solve. Each iteration takes
$s=-HF$, evaluates the residual change $y$, and applies the rank-one update

$$
H\leftarrow H+\frac{(s-Hy)s^T H}{s^T H y}.
$$

Updates with tiny secant denominators are skipped. Subsequent iterations need
residual evaluations and matrix-vector products, not new Jacobians or matrix
factorizations. The final residual must pass `check`; a failed solve is never
silently accepted. This $(q,p)$ map is second order and symplectic where the
potential is smooth, up to the nonlinear solve error. It
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
from coursekit.labs.line_following import check, compile_step, initial_state, robot, step

torch.set_num_threads(1)
model = robot()
state = initial_state(model)
action = state.q.new_tensor([0.2, 0.2])
advance = compile_step()
for _ in range(1000):
    state, diagnostics = advance(state, action, model)
    check(state, diagnostics)
```

Use `advance = step` to inspect ordinary Python execution. `compile_step` compiles
three kernels: resistance, the conservative solve, and terminal-state handling.
Both half steps reuse the same resistance kernel. Compilation needs PyTorch 2.14
or later and its supported compiler
toolchain; the first call has a substantial cost. Construct `advance` once,
outside the rollout loop.

For stacked states and actions, use `compile_step(batched=True)`. Add
`model_batched=True` when each robot has its own stacked parameter set. The eager
equivalents are `torch.vmap(step, in_dims=(0, 0, None))` for shared parameters and
`in_dims=(0, 0, 0)` for per-robot parameters.

Compilation specializes on shapes, dtype, device, and Python settings such as
`dt`. A new batch size creates new compiled variants; revisiting a cached size
reuses them. Keep the environment count fixed during training rollouts. Changing
tensor values, including actions and robot parameters, does not require new
variants when shapes and layouts stay the same. Successive steps and both
resistance half steps use compatible layouts. Dynamic batch sizes are not
supported by this compilation helper.

Do not put these calculations in `torch.inference_mode`, which disables the
forward differentiation used for kinematics. A complete rollout is not a smooth
function at impacts, friction
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
terminal falls, batching, and compiler graph capture and reuse.
