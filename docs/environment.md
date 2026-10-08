# Python environment

Python 3.12.14 and dependencies are frozen by `.python-version` and `uv.lock`.
Run commands with `uv run --locked` from the
[project directory](../README.md#start-here), or use the numbered wrappers. The first
run prepares the environment automatically.
The environment includes NumPy, SciPy, scikit-learn, joblib, Pillow, Matplotlib,
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

Shared **pilot** limits: 2 CPU quota, 2048 MiB memory, 128 processes, one BLAS/OpenMP
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
120 **simulated seconds** allowed for an out-and-back journey in the pilot task.
Only the documented observation tensors and reset mask reach its controller.

For short robot debugging runs, construct `Environment(..., compiled=False)`;
normal evaluation compiles the same physics. Keep the environment alive and reuse
its batch size for longer experiments. Autograd is available inside `act`, including
during instructor evaluation. Do not put the simulator in `torch.inference_mode()`:
its mechanics uses automatic differentiation internally.

`view_lab` adds the optional locked PySide6 Essentials and ModernGL packages.
The robot viewer requires a desktop display and OpenGL 3.3 or newer. The simulator,
training, evaluation and Blindjack's HTML viewer do not require an OpenGL context.
