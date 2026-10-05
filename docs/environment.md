# Python environment

Python 3.12.14 and dependencies are frozen by `.python-version` and `uv.lock`.
Run commands with `uv run --locked` from the student project root, or use the
numbered wrappers. The first run prepares the environment automatically.
The environment includes NumPy, SciPy, scikit-learn, joblib, Pillow, Matplotlib,
CPU PyTorch, and marimo for the [development notebook](notebook.md).

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

Current **pilot** limits: 2 CPU quota, 2048 MiB memory, 128 processes, one BLAS/OpenMP
thread, 60 seconds initialization, 5 seconds per hand, 300 seconds total, 64 KiB
logs, and 256 MiB temporary storage. ZIPs are limited to 256 MiB compressed,
512 MiB expanded, and 4096 files. These grading limits do not constrain your local
training or analysis.

For Line Following, the per-request limit is 1 second, and the 300-second total
applies to initialization plus controller request wall time for one episode.
Host simulation and compilation do not use that controller budget. Each assessed
episode gets a fresh controller process; memory persists through the outward and
return legs. Only the documented observation tensors and reset mask are supplied.

`view_lab` adds the optional locked PySide6 Essentials and ModernGL packages.
The robot viewer requires a desktop display and OpenGL 3.3 or newer. The simulator,
training, evaluation and Blindjack's HTML viewer do not require an OpenGL context.
