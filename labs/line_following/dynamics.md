# Robot dynamics

This guide explains how the robot's geometry and energy define its equations of
motion, how contacts and motors act on those equations, and how the simulator
advances them in time. Read the [task and episode rules](README.md) first. For
feedback control, state estimation and learning methods, see the
[control and learning guide](control_hints.md).

The working simulator uses [analytic Numba mechanics](../../src/coursekit/labs/line_following_numba.py)
and a [native integration loop](../../src/coursekit/labs/line_following_rollout.py).
Training, the graphical viewer and instructor evaluation use this same engine.
The [PyTorch autodiff example](../../src/coursekit/labs/line_following.py) derives the
same mechanics from energy and provides an independent check of the analytic
formulas. Its optional `torch.compile` acceleration is not needed to run the lab;
see the [installation instructions](../../docs/environment.md#optional-acceleration-of-the-pytorch-mechanics-example).
The current robot and numerical settings remain a pilot profile; assessment
parameter ranges still require validation. Course geometry, observations, route
coverage, parameter sampling and scoring live in adjacent public modules; see the
[lab statement and environment API](README.md).

## Coordinates and energy

Use SI units and float64 arrays (tensors in the autodiff example). The world has a horizontal floor at $z=0$.
The body frame has $x$ forward, $y$ left, and $z$ up. The configuration is

$$
q=(x,y,z,\phi,\theta,\psi,\alpha_L,\alpha_R).
$$

The first three coordinates locate the axle midpoint. Body rotation is
$R=R_z(\psi)R_y(\theta)R_x(\phi)$. Wheel angles are relative to the body.
Positive wheel rotation drives forward; positive pitch leans forward. The sphere
centre is fixed behind the axle. Its orientation is irrelevant because it is
homogeneous and isotropic; its world angular momentum is a separate state.

Each centre has position $c_i=x+Rr_i$. In the reference, `torch.func.jvp` differentiates this
pose map to obtain translational velocities and $\dot R$; angular velocity is
read from $\dot R R^T$. Add wheel spin about the axle to obtain each wheel's
angular velocity. Translational and rotational kinetic energies give

$$
L(q,v)=T(q,v)-V(q),\qquad
p=\partial_v L,\qquad M(q)=\partial_v^2 L,\qquad v=M(q)^{-1}p.
$$

Here $T$ is kinetic energy and $V$ contains gravity and contact elasticity.
The gravitational contribution is $g\sum_i m_i c_{i,z}$. The momentum $p$ is
conjugate to the configuration coordinates, rather than a vector of Cartesian
velocities. For these coordinates the Hamiltonian is

$$
H(q,p,s)=\tfrac12 p^T M(q)^{-1}p+V(q)+\frac{\lVert s\rVert^2}{2I_b},
$$

where $s$ is the sphere's angular momentum and $I_b$ its scalar rotational inertia.
The integrator below evolves $(q,p)$ using the equivalent discrete Lagrangian
formulation; the sphere's rotation and dissipative interactions are handled separately.

The autodiff example avoids the velocity Hessian. Stack linear and angular
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
At fixed pose, velocity and both contact responses share one inertia solve
with multiple right-hand sides, so the matrix is factorized only once.

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

The environment requires normalised motor actions in $[-1,1]$ and rejects invalid
commands. The motor law then limits the resulting torque:

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

## Analytic implementation and native integration

The Numba engine evaluates the same energy derivatives explicitly. Let $E$ map
Euler-angle rates to body angular velocity, and $W=RE$ map them to world angular
velocity. A centre offset $c=Rr$ has velocity
$u=\dot x+\omega\times c$ and angle derivative
$\partial_a c=W_{:,a}\times c$. For the energy,

$$
\partial_a T=\sum_i m_i u_i\cdot\partial_a u_i
+\sum_i(I_i\omega_i)\cdot\partial_a\omega_i.
$$

`lagrangian_derivatives` contracts these terms directly instead of constructing
a third-order array of mass derivatives. Velocity derivatives give the generalized
momenta; gravity and spring derivatives supply the potential contribution.
`kinematics`, `inertia` and `contacts` provide the centre/angular maps, positive
mass matrix and material contact velocities. `resistance`, `conservative_step`
and `step` implement the physical substeps described above. These are ordinary,
small Python functions compiled with Numba; no finite-difference derivatives or
coordinate Hessians are used.

The working environment calls a native block of 20 physical steps per controller
action. Every 1 ms step still checks the numerical residual, updates certified
sensor-row coverage and path length, and handles turnaround, fall and completion.
Course queries use the same cubic controls, quadrature and curve-error bounds as
the reference. Coverage follows only the connected interval beginning at the
accepted frontier, stopping at the first uncovered gap. This ordering avoids
searching and sorting unrelated branches at every step.

The controller is called in Python between blocks, with the original PyTorch
observation/action API. Optional block snapshots retain every physical frame for
the viewer; playback does not rerun physics or controller inference. Noise is
drawn from each episode's own random generator after the block, preserving
independence from batching and unrelated resets.

Rows are independent; each row's time steps remain sequential. Batches of eight
or more robots use a native parallel loop with at most four CPU threads (or the
smaller Numba thread limit). Smaller batches use a serial native loop. The engine
restores the caller's thread setting after every block. Larger pools increased
overhead for these small matrix problems in the development measurements.

The first execution compiles Numba functions. Their array types, ranks and layouts
are specialized, but array lengths and robot parameter values are runtime data.
A different batch length therefore reuses the functions with those same types.
The native cache lives under `coursekit/numba/` in the user cache directory;
`NUMBA_CACHE_DIR` or `COURSEKIT_COMPILE_CACHE` can supply its root. Its directory
identifies the complete engine sources and Python/Numba/LLVM/NumPy/SciPy versions, and Numba checks
CPU compatibility and writes artifacts atomically. Source-wide invalidation is
needed because Numba does not itself track changes in imported helpers. Cache
files contain executable code: keep them local, outside submission archives.
The selected cache directory must be writable. Set `COURSEKIT_COMPILE_CACHE` to
a writable local directory when the normal user cache is unavailable. Engine
functions do not fall back to an unversioned cache alongside the source files.

## Running and inspecting the core

The ordinary `Environment(...)` API uses Numba and requires no C++ toolchain.
Use `Environment(..., compiled=False)` to inspect the slower autodiff reference
with the same observations and episode rules. The following example explores
that reference directly; run it in `uv run --locked python` from the project
directory containing `pyproject.toml` and `uv.lock`:

```python
import torch
from coursekit.labs.line_following import check, initial_state, robot, step

model = robot()
state = initial_state(model)
action = state.q.new_tensor([0.2, 0.2])
for _ in range(10):
    state, diagnostics = step(state, action, model)
    check(state, diagnostics)
```

To study compiler acceleration, install the [optional system tools](../../docs/environment.md#optional-acceleration-of-the-pytorch-mechanics-example),
then construct a compiled function once, outside the rollout loop:

```python
from coursekit.labs.line_following import compile_step

torch.set_num_threads(1)
advance = compile_step()
state, diagnostics = advance(state, action, model)  # includes initial compilation
check(state, diagnostics)
```

`compile_step` compiles resistance, the conservative solve and terminal handling
as three kernels; both resistance half steps reuse one kernel. The C++ call
wrapper reduces Python overhead for small matrices. Linear solves use
`torch.linalg.solve_ex` and return non-finite values on failure; `check` raises
outside the graph, preserving compiler caching. Both discrete momenta are
evaluated together to share derivative work.

For experiments requiring persistent PyTorch artifacts, the optional
`line_following_cache.ahead` helper uses PyTorch's experimental
[`aot_compile` interface](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_aot_compile.html).
It retains runtime guards and keys artifacts by engine source, runtime, CPU,
batch size and thread count. Writes are locked and atomic; incompatible or
damaged artifacts are rebuilt. This helper belongs to the PyTorch example;
the working Numba environment does not load or create these artifacts.

For stacked states/actions use `compile_step(batched=True)`, adding
`model_batched=True` for independent parameter sets. The eager equivalents use
`torch.vmap(step, in_dims=(0, 0, None))` or `in_dims=(0, 0, 0)`. This optional
PyTorch helper specializes on shapes, dtype, device and Python settings such as
`dt`: new batch sizes can compile again, while changed values with matching
layouts reuse a specialization. First-call time must be measured separately
from steady throughput.

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
The [native tests](../../tests/test_line_following_numba.py) independently check
momentum conservation, passive losses, friction bounds and time-step convergence;
they also compare analytic derivatives, contact maps and trajectories with the
autodiff implementation. Environment tests execute saved native kernels in a fresh
process with compilation forbidden, changed robot parameters and different batch
lengths, including the parallel path.
