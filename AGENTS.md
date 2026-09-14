# Student project

Write solutions in `work/<lab>/<variant>/`. Course updates own tracked files;
never store a solution, dataset, or model in those files. Use the shared locked
environment. Keep code short and explain the mathematical idea and invariants.
Include reproducible training, all inference artifacts, and an honest AI-use
declaration. Inference must run on CPU without network access.

Images are stored once in lossless PNG sheets. Other numeric data use compressed
NPZ with pickling disabled. Preserve image indices and labels when loading sheets.
Run `uv run --locked pytest` for public package tests.
