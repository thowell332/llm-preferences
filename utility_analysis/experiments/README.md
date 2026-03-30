This folder contains experiment scripts referenced by `utility_analysis/experiments.yaml`.

## Linear probes (`linear_probes`)

`experiments/linear_probes/run_linear_probes.py` supports two stages:

- `--stage collect`: for each `(role, option)` prompt, greedily generates the rating and stores:
  - the parsed rating (1–10 when parsable)
  - residual-stream activations at:
    - the **last prompt token**
    - the **first generated token**
- `--stage train`: fits **per-layer** ridge-regularized linear probes to predict either:
  - `--target utility` (default), or
  - `--target rating` (filters unparseable ratings)

`experiments.yaml` includes:

- `linear_probes_collect`
- `linear_probes_train`

You will usually run `collect` first, then `train`, pointing `utilities_path` at a precomputed utilities JSON aligned by option id.