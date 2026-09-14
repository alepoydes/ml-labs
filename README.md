# Machine learning course

Five substantial labs connect machine learning methods with complete working
solutions. Blindjack is available first; [four more labs](docs/planned-labs.md)
are planned.

## Start here

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone the
student repository supplied by your instructor, and enter its root directory.
Commands use one shared locked Python environment.

```sh
./step_0_update_resources
./step_1_new_solution blindjack baseline --template baseline
./step_2_train work/blindjack/baseline
./step_3_evaluate work/blindjack/baseline
```

Read [the Blindjack statement](labs/blindjack/README.md), then inspect the baseline's
`train.py`, `solution.py`, and configuration. Open `data/blindjack/train/images-0000.png`
in an ordinary image viewer to inspect the training data.

Create your own variant from the minimal starter:

```sh
./step_1_new_solution blindjack my-model
./step_2_train work/blindjack/my-model
./step_3_evaluate work/blindjack/my-model
./step_4_pack work/blindjack/my-model
./step_5_check_submission dist/blindjack-my-model-<hash>.zip
```

Replace `<hash>` with the filename printed by packaging. Edit and repeat training
and evaluation as needed. The final check requires the [reference container](docs/environment.md).
It extracts a fresh copy and finds dependencies on files outside your submission.
Submit the resulting ZIP through the instructor's designated channel; packaging
does not upload anything.

On Windows use, for example, `.\step_2_train.cmd work\blindjack\my-model`.
All paths are relative to the
student project root, or absolute. The wrappers use their own directory even when
called from somewhere else.

## Course updates and ownership

```sh
git pull --ff-only
./step_0_update_resources
```

The resource updater does not pull Git changes. It validates SHA-256 hashes, replaces
changed resources, and removes obsolete course-managed files. Interrupted updates
must be rerun before training or evaluation.

Keep your variants and models under `work/`, experiments and logs under `runs/`, and
packaged archives under `dist/`. These directories are ignored by Git and reserved
for your work. `data/` is managed course storage. Updating a template never changes
an existing variant. A new-solution command refuses to overwrite it.

See [submission format](docs/submissions.md), [environment](docs/environment.md), and
[AI use](docs/ai-use.md). If resources have not yet been published, ask the instructor
for the release rather than inventing download URLs.
