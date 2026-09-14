# Machine learning course

Five substantial labs connect machine learning methods with complete working
solutions. Blindjack is available first; [four more labs](docs/planned-labs.md)
are planned.

## Start here

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone the
student repository supplied by your instructor, and enter its root directory.
Commands use one shared locked Python environment.

`labs/blindjack/` provides two templates: `starter` is a minimal example of the
solution format; `baseline` is a naive, trainable solution. To run either one,
first copy it into `work/` with `step_1_new_solution`. Edit, train, and evaluate
that working copy. Leave the tracked course files, including `labs/`, unchanged
so course updates do not conflict with your changes.

```sh
./step_0_update_resources
./step_1_new_solution blindjack --template baseline
./step_2_train work/blindjack/baseline
./step_3_evaluate work/blindjack/baseline
```

Read [the Blindjack statement](labs/blindjack/README.md), then inspect `train.py`,
`solution.py`, and `config.toml` in `work/blindjack/baseline/`.
Open `data/blindjack/index.html` in a browser to explore labeled examples and try
the validation hands yourself while the model trains. It works offline, without
Python or a server. You can also open `train/labels.csv` and the PNG sheets under
`data/blindjack/` directly; CSV coordinates count rows and columns from 1.

Create your own variant from the minimal starter:

```sh
./step_1_new_solution blindjack my-model
./step_2_train work/blindjack/my-model
./step_3_evaluate work/blindjack/my-model
./step_4_pack work/blindjack/my-model
./step_5_check_submission dist/blindjack-my-model-<hash>.zip
```

Replace `<hash>` with the filename printed by packaging. `step_3_evaluate` tests
the working directory on public validation data; `step_5_check_submission` tests
the packaged ZIP. Edit and repeat training and evaluation as needed. The final
check requires the [reference container](docs/environment.md).
It extracts a fresh copy and finds dependencies on files outside your submission.
Submit the resulting ZIP from `dist/` through the instructor's designated channel.
Do not submit the template in `labs/` or the working directory itself; packaging
does not upload anything.

On Windows use, for example, `.\step_2_train.cmd work\blindjack\my-model`.
All paths are relative to the student project root, or absolute. The wrappers use
their own directory even when called from somewhere else.

## Copy a template

The copy keeps the selected template's name unless you supply a new name.
`--template` chooses the source; the optional name chooses the destination.
Choose the command for the copy you want; an existing directory is never overwritten.

```sh
# starter -> work/blindjack/starter
./step_1_new_solution blindjack

# baseline -> work/blindjack/baseline
./step_1_new_solution blindjack --template baseline

# baseline -> work/blindjack/my-model
./step_1_new_solution blindjack my-model --template baseline
```

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
