This folder contains experiment scripts referenced by `utility_analysis/experiments.yaml`.

## Linear probes (`linear_probes`)

Entry point: `experiments/linear_probes/run_linear_probes.py` (thin wrapper). For the **notebook-style pilot** (sample layers → collect → train → optional plot) in one command, use:

```bash
# From repo root: activate your venv first (see utility_analysis/README.md).
source venv/bin/activate

cd utility_analysis/experiments/linear_probes
python run_pilot_sweep.py --model-key llama-31-8b-instruct
```

Otherwise run `collect` then `train` yourself (two commands). Implementation modules live in `experiments/linear_probes/lp/`:

| Module | Role |
|--------|------|
| `lp/cli.py` | Argument parser and `main()` |
| `lp/data.py` | Options/utilities/roles, `models.yaml`, prompts, `ExampleMeta` |
| `lp/activations.py` | HF hooks + vLLM hidden-state extraction helpers |
| `lp/hf_loader.py` | Hugging Face `from_pretrained` / device-map / bitsandbytes |
| `lp/collect.py` | Collect stage orchestration (`collect_hf` / `collect_vllm`) |
| `lp/train.py` | Train stage (ridge probes) |
| `lp/metrics.py` | Ridge / Spearman / R² |
| `lp/debug.py` | RSS logging, path warnings |

The script supports two stages:

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

**WSL / limited RAM:** the default HF path stages the full fp16 weights on CPU (~18+ GiB RAM) before moving to GPU; the process may be **SIGKILL’d (exit -9)** while loading shards. Use **`--hf_bnb_8bit`** (requires `bitsandbytes` in `requirements.txt`) for 8-bit GPU loading, or raise WSL memory and keep full-precision staging.