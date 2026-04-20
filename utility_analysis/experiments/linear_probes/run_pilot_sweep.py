#!/usr/bin/env python3
"""
Run the linear-probes pilot sweep from the shell (same flow as the notebook).

1. Load models.yaml, infer num_hidden_layers, build a layer index list (sample or all).
2. run_linear_probes.py --stage collect ...
3. run_linear_probes.py --stage train ...
4. Optionally plot test metric vs layer (PNG, headless-friendly).

Run from anywhere; paths default to the repo layout under utility_analysis/.

Example (activate the repo venv first: ``source venv/bin/activate`` from the repo root):

  cd utility_analysis/experiments/linear_probes
  python run_pilot_sweep.py --model-key llama-31-8b-instruct

Regenerate the plot from an existing probe JSON (no collect/train):

  python run_pilot_sweep.py --model-key llama-31-8b-instruct --plot-only

Or with an explicit JSON path (relative to ``linear_probes/`` or absolute):

  python run_pilot_sweep.py --model-key llama-31-8b-instruct --plot-only \\
    --probe-results-json results_linear_probes/llama-31-8b-instruct/linear_probes_...json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from transformers import AutoConfig

_LP_DIR = Path(__file__).resolve().parent
_UTILITY_ANALYSIS = _LP_DIR.parent.parent
_RUN_LINEAR_PROBES = _LP_DIR / "run_linear_probes.py"


def _default_utilities_path(model_key: str, role: str) -> str:
    role_slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", role).strip("_").lower() or "role"
    return (
        f"../../shared_utilities/options_custom/{model_key}/"
        f"results_utilities_{model_key}_{role_slug}.json"
    )


def _layer_indices(layer_mode: str, num_layers: int, num_sampled: int) -> list[int]:
    if layer_mode == "all":
        return list(range(num_layers))
    if layer_mode == "sample":
        sampled = np.linspace(0, num_layers - 1, num=num_sampled)
        return sorted({int(round(x)) for x in sampled})
    raise ValueError(f"Unknown layer mode: {layer_mode!r} (use 'all' or 'sample')")


def _run_linear_probes(argv: list[str], env: dict[str, str]) -> None:
    cmd = [sys.executable, "-u", str(_RUN_LINEAR_PROBES), *argv]
    print("Running:", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=str(_LP_DIR), env=env)
    if rc.returncode != 0:
        raise SystemExit(rc.returncode)


def main() -> None:
    p = argparse.ArgumentParser(description="Linear probes pilot sweep (collect + train + optional plot).")
    p.add_argument("--model-key", required=True, help="Key in utility_analysis/models.yaml")
    p.add_argument("--role", default="helpful assistant", help="Single role string for collect --roles")
    p.add_argument(
        "--options-path",
        default="../../shared_options/options_custom.json",
        help="Path relative to experiments/linear_probes/",
    )
    p.add_argument(
        "--utilities-path",
        default=None,
        help="Relative to experiments/linear_probes/; default derived from model key and role",
    )
    p.add_argument(
        "--save-dir",
        default=None,
        help="Under utility_analysis/; default results_linear_probes/<model-key>",
    )
    p.add_argument("--layer-mode", choices=["sample", "all"], default="sample")
    p.add_argument("--num-sampled-layers", type=int, default=10)
    p.add_argument("--max-examples", type=int, default=20)
    p.add_argument("--max-new-tokens-for-parsing", type=int, default=3)
    p.add_argument("--max-model-len", type=int, default=256)
    p.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    p.add_argument(
        "--vllm-no-compile",
        action="store_true",
        help="Forward to collect: vLLM compilation_config=0 (often helps flaky WSL/CUDA setups).",
    )
    p.add_argument(
        "--vllm-attention-backend",
        default=None,
        help="Forward to collect, e.g. flash_attn to avoid FlashInfer.",
    )
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument(
        "--no-hf-bnb-8bit",
        action="store_true",
        help="Disable bitsandbytes 8-bit HF load (on by default for --backend hf; matches the notebook pilot).",
    )
    p.add_argument("--position", choices=["prompt_last", "gen_first"], default="gen_first")
    p.add_argument("--target", choices=["utility", "rating"], default="utility")
    p.add_argument("--probe-mode", choices=["all", "per_role", "cross_role"], default="all")
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip collect and train; only read existing probe metrics JSON and save the figure. "
        "Locates JSON from --model-key, --role, --save-dir, --position, --target, --probe-mode unless "
        "--probe-results-json is set (path relative to experiments/linear_probes/ or absolute).",
    )
    p.add_argument(
        "--probe-results-json",
        default=None,
        help="With --plot-only: explicit path to probe results JSON (relative to experiments/linear_probes/ if not absolute).",
    )
    p.add_argument("--test-fraction", type=float, default=0.3)
    p.add_argument("--ridge-lambda", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--plot-path",
        default=None,
        help="If set, save metric-vs-layer plot to this path (PNG). Default: <save-dir>/pilot_sweep_<suffix>.png",
    )
    p.add_argument("--no-plot", action="store_true", help="Skip matplotlib plot")

    args = p.parse_args()

    save_dir_rel = args.save_dir or f"results_linear_probes/{args.model_key}"
    save_suffix = f"{args.model_key}_pilot_{args.role}".replace(" ", "_")

    if args.plot_only:
        from notebook_runs import best_layers_summary, existing_probe_results_path, plot_probe_results_file

        results_path = existing_probe_results_path(
            _UTILITY_ANALYSIS.parent,
            save_dir=save_dir_rel,
            save_suffix=save_suffix,
            position=args.position,
            target=args.target,
            probe_mode=args.probe_mode,
            explicit_path=args.probe_results_json,
        )
        if not results_path.is_file():
            sys.exit(f"--plot-only: missing probe results JSON: {results_path}")

        summ = best_layers_summary(results_path)
        pm = summ["primary_metric"]
        print(f"[plot-only] {results_path}")
        print(f"Best layer by {pm} (max): {summ['best_layer_primary']}")
        print(f"Best layer by test MSE (min): {summ['best_layer_mse']}")

        if args.no_plot:
            return

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping plot.", file=sys.stderr)
            return

        plot_path = args.plot_path
        if plot_path is None:
            plot_path = str((_LP_DIR / save_dir_rel / f"pilot_sweep_{save_suffix}.png").resolve())
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)

        tit = f"Pilot sweep ({args.target}, {args.position}, role={args.role}) [plot-only]"
        fig, _, _ = plot_probe_results_file(results_path, title=tit)
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved plot:", plot_path)
        return

    models_path = _UTILITY_ANALYSIS / "models.yaml"
    with open(models_path, "r", encoding="utf-8") as f:
        models_cfg = yaml.safe_load(f)
    if args.model_key not in models_cfg:
        sys.exit(f"Unknown model key {args.model_key!r} in {models_path}")

    model_entry = models_cfg[args.model_key]
    model_path = model_entry.get("path") or model_entry.get("model_name")
    if not model_path:
        sys.exit(f"No path/model_name for {args.model_key!r} in models.yaml")

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = int(getattr(cfg, "num_hidden_layers"))
    pilot_layers = _layer_indices(args.layer_mode, num_layers, args.num_sampled_layers)
    layers_spec = ",".join(str(x) for x in pilot_layers)

    utilities_rel = args.utilities_path or _default_utilities_path(args.model_key, args.role)

    options_abs = (_LP_DIR / args.options_path).resolve()
    utils_abs = (_LP_DIR / utilities_rel).resolve()
    if not options_abs.is_file():
        sys.exit(f"Missing options file: {options_abs}")
    if not utils_abs.is_file():
        sys.exit(f"Missing utilities file: {utils_abs}")

    print("Pilot role:", args.role)
    print("Model layers:", num_layers)
    print("Pilot layers:", pilot_layers)
    print("Layers spec:", layers_spec)
    print("Options:", options_abs)
    print("Utilities:", utils_abs)

    child_env = {**os.environ, "PYTHONFAULTHANDLER": "1"}
    if args.backend == "hf":
        child_env["CUDA_LAUNCH_BLOCKING"] = "1"

    collect_argv = [
        "--model_key",
        args.model_key,
        "--stage",
        "collect",
        "--backend",
        args.backend,
        "--save_dir",
        save_dir_rel,
        "--save_suffix",
        save_suffix,
        "--options_path",
        args.options_path,
        "--utilities_path",
        utilities_rel,
        "--roles",
        args.role,
        "--layers",
        layers_spec,
        "--max_new_tokens_for_parsing",
        str(args.max_new_tokens_for_parsing),
        "--max_model_len",
        str(args.max_model_len),
        "--max_examples",
        str(args.max_examples),
        "--trust_remote_code",
    ]
    if args.backend == "hf":
        collect_argv.extend(
            [
                "--fp16",
                "--cuda_launch_blocking",
                "--attn_implementation",
                "eager",
            ]
        )
    if args.force_cpu:
        collect_argv.append("--force_cpu")
    if args.backend == "hf" and not args.no_hf_bnb_8bit:
        collect_argv.append("--hf_bnb_8bit")
    if args.backend == "vllm":
        if args.vllm_no_compile:
            collect_argv.append("--vllm-no-compile")
        if args.vllm_attention_backend:
            collect_argv.extend(["--vllm-attention-backend", args.vllm_attention_backend])

    _run_linear_probes(collect_argv, child_env)

    train_argv = [
        "--model_key",
        args.model_key,
        "--stage",
        "train",
        "--save_dir",
        save_dir_rel,
        "--save_suffix",
        save_suffix,
        "--position",
        args.position,
        "--target",
        args.target,
        "--probe_mode",
        args.probe_mode,
        "--test_fraction",
        str(args.test_fraction),
        "--ridge_lambda",
        str(args.ridge_lambda),
        "--seed",
        str(args.seed),
    ]
    _run_linear_probes(train_argv, child_env)

    # save_dir is resolved relative to experiments/linear_probes/ (same cwd as run_linear_probes.py).
    results_path = (
        _LP_DIR / save_dir_rel / f"linear_probes_{save_suffix}_probe_results_{args.position}_{args.target}_{args.probe_mode}.json"
    )
    if not results_path.is_file():
        print(f"Warning: expected results file missing: {results_path}", file=sys.stderr)
        return

    from notebook_runs import best_layers_summary, plot_probe_results_file

    summ = best_layers_summary(results_path)
    pm = summ["primary_metric"]
    print(
        f"Pilot best layer by {pm} (max): {summ['best_layer_primary']} "
        f"(see probe JSON for value at that layer)"
    )
    print(f"Pilot best layer by test MSE (min): {summ['best_layer_mse']}")

    if args.no_plot:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.", file=sys.stderr)
        return

    plot_path = args.plot_path
    if plot_path is None:
        plot_path = str((_LP_DIR / save_dir_rel / f"pilot_sweep_{save_suffix}.png").resolve())
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)

    tit = f"Pilot sweep ({args.target}, {args.position}, role={args.role})"
    fig, _, _ = plot_probe_results_file(results_path, title=tit)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved plot:", plot_path)


if __name__ == "__main__":
    main()
