# Environment and archive checks

Python 3.12.14 and dependencies are frozen by `.python-version` and `uv.lock`.
Training and local evaluation use `uv run --locked` in this project. The environment
includes NumPy, SciPy, scikit-learn, joblib, Pillow, Matplotlib, and CPU PyTorch.
It also includes marimo for the [baseline development notebook](notebook.md).
Per-submission dependency installation and network services are unsupported.

Build the reference image from the student project root:

```sh
docker build -t mlcourse-reference:local .
```

Podman uses the equivalent command. Set `COURSE_CONTAINER_ENGINE` to select one
explicitly if both are installed. Linux with functioning cgroup CPU, memory and
process limits is the reference host. Docker Desktop supplies a Linux environment
on Windows/macOS. Windows wrapper syntax is `.\step_5_check_submission.cmd ...`.

The check starts a fresh container, mounts only your extracted submission read-only,
disables networking, and supplies temporary writable storage. It loads the model
once and receives one seven-image observation at a time. Logs are separate from
protocol output; scores are computed outside the worker.

Current **pilot** limits: 2 CPU quota, 2048 MiB memory, 128 processes, one BLAS/OpenMP
thread, 60 seconds initialization, 5 seconds per hand, 300 seconds total, 64 KiB
logs, 256 MiB temporary storage. ZIPs are limited to 256 MiB compressed, 512 MiB
expanded, and 4096 files; selected model artifacts share that total limit.
The exact host and image ID are part of the instructor's evaluation context.

Invalid archives, unsupported APIs, model-loading failures, invalid masks, runtime
errors, protocol errors, timeouts, output limits, and memory/process failures are
reported as failed attempts with no partial score. Infrastructure failures are
retryable and are never silently scored zero. Local working-directory evaluation
is convenient during development but is not isolated; use it only on your own code.
Evaluation and archive checks print the original worker traceback on failure and
save bounded logs under `runs/<lab>/<variant>/evaluate.log` and `check.log`.

The two-machine comparison and Windows execution checks remain release validation
tasks; the primary development checks run on a Linux CPU host.
