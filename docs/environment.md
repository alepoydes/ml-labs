# Python environment

Python 3.12.14 and dependencies are frozen by `.python-version` and `uv.lock`.
Run commands with `uv run --locked` from the
[project directory](../README.md#start-here), or use the numbered wrappers. The first
run prepares the environment automatically.
The environment includes NumPy, Numba, SciPy, scikit-learn, joblib, Pillow, Matplotlib,
CPU PyTorch, and marimo. Both labs use this environment; the supplied
[development notebook](notebook.md) belongs to the Blindjack baseline.

Training, notebooks, evaluation, and ZIP checks run directly in your Python
environment. You do not need Docker or Podman. There are no course-imposed network,
filesystem, time, or memory restrictions on these local runs. Prints and exceptions
behave as in any Python program, and you can use an ordinary editor debugger.
Choose this project's `.venv/bin/python` as its interpreter (`.venv\Scripts\python.exe`
on Windows). See [developing a solution](development.md) for direct Python examples.

`step_5_check_submission` validates a ZIP, extracts a fresh copy, and evaluates it
in the same Python environment. This catches missing packaged files when your code
uses paths relative to the supplied `artifact_dir`. It does not block access to
files elsewhere on your computer or reproduce instructor resource limits.
Run local commands only on code and models you trust.

## Requirements for submitted inference

The instructor evaluates submissions separately with protections. Final inference
must use the locked libraries, run on CPU without network services, and read model
files relative to `artifact_dir`. The submission directory is read-only; temporary
files may be written under `/tmp`. Only submitted files and the installed environment
are available. Dependencies cannot be installed during grading.

Shared **pilot** limits: 2 CPU quota, 2048 MiB memory, 128 processes, one BLAS/OpenMP/Numba
thread, 60 seconds initialization, 64 KiB logs, and 256 MiB temporary storage.
ZIPs are limited to 256 MiB compressed,
512 MiB expanded, and 4096 files. These grading limits do not constrain your local
training or analysis.

| Limit or lifecycle | Blindjack | Line Following |
| --- | --- | --- |
| Acting method | `select(images)` | `act(observation, reset)` |
| Request wall time | 5 seconds per hand | 1 second per control request |
| Total controller wall time | 300 seconds for the hand set | 300 seconds per episode |
| Controller lifetime | One instance for the hand set | A fresh process per assessed episode; memory persists through both legs |

The robot's total controller budget includes initialization and request wall time.
Host simulation and compilation do not use that budget. This is separate from the
case-specific **simulated time** limit of `30 * T_100` for the complete
out-and-back journey. `T_100` is the mechanical reference time for that course and robot.
Only the documented observation tensors and reset mask reach its controller.

Line Following uses the public Numba CPU engine for training and evaluation.
The locked Numba and llvmlite binary packages include the required LLVM components:
normal use does not require a separately installed C++ compiler, CUDA or JAX.
Use a platform with binary wheels for the locked packages; do not replace a missing
LLVM wheel with an improvised source build. The current simulator and desktop
workflow have been exercised on Linux x86-64; Windows and macOS still need native
platform validation.
The first run compiles native functions; later processes reuse a local disk cache.
An empty-cache Linux measurement took about 57 seconds for environment setup and
the first control interval; this is still a substantial first-run cost. A prepared
single-robot environment then simulated one second in about 0.10 seconds.
These are measurements on one development machine, not hardware-independent
guarantees. Keep the cache between runs and allow the viewer's preparation to finish.
Changing the number of robots does not require a new specialization when array
types and layouts remain unchanged. See the [mechanics guide](../labs/line_following/dynamics.md).

`Environment(..., compiled=False)` runs the slower PyTorch autodiff reference for
study and comparison. It needs no C++ compiler either. Autograd is available inside
`act`, including during instructor evaluation. Do not put the autodiff reference
in `torch.inference_mode()`: its kinematics needs forward differentiation.

`view_lab` adds the optional locked PySide6 Essentials and ModernGL packages.
The robot viewer requires a desktop display and OpenGL 3.3 or newer. The simulator,
training, evaluation and Blindjack's HTML viewer do not require an OpenGL context.

## Optional acceleration of the PyTorch mechanics example

The [autodiff example](../labs/line_following/dynamics.md#running-and-inspecting-the-core)
also exposes `compile_step()`, which uses `torch.compile` with CPU TorchInductor.
This is an optional experiment; the normal robot environment and viewer use Numba.
PyTorch itself is already installed. Its external C++ compiler and development
tools are **optional system dependencies**, not prerequisites for the lab.

PyTorch's [compilation tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
requires a C++ compiler such as `g++` and Python development headers. On Linux,
install the compiler through your distribution's package manager (on Debian/Ubuntu,
`sudo apt install build-essential`). Check `g++ --version`. Headers must belong to
the interpreter used by `uv`, rather than an unrelated system Python; inspect them with:

```sh
uv run --locked python -c "import sysconfig; from pathlib import Path; p = Path(sysconfig.get_path('include')) / 'Python.h'; print(p, p.is_file())"
```

On Windows, follow the [PyTorch CPU installation instructions](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html):
install MSVC with **Desktop development with C++**, then run commands from an
environment initialized by its `vcvars64.bat` (or the corresponding x64 developer
command prompt). LLVM's `clang-cl` is an alternative, but that guide still requires
the MSVC runtime libraries. LLVM alone does not replace this setup. The optional
compiled mechanics example has not been validated on macOS; the normal Numba
environment does not depend on that optional toolchain.

Check actual CPU compilation before using the example:

```sh
uv run --locked python -c "import torch; f = torch.compile(lambda x: x.sin() + x.square(), fullgraph=True); print(f(torch.arange(4, dtype=torch.float64)))"
```

Its first execution includes compilation. Measure later calls separately and keep
the compiled function alive. CPU Inductor specializes on tensor shapes and may
compile again for another batch size. These requirements concern the optional
PyTorch example, not the Numba engine. Submitted controllers must remain runnable
in the supplied inference environment without installing external build tools.
