from __future__ import annotations

import argparse
import os

from lp.collect import collect
from lp.data import none_or_str
from lp.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear probes: collect activations and train per-layer probes.")
    parser.add_argument("--model_key", default="llama-31-8b-instruct", help="Model key from models.yaml")
    parser.add_argument("--save_dir", default="results/<model_key>", help="Directory to save outputs")
    parser.add_argument("--save_suffix", type=none_or_str, default=None, help="Custom suffix for saved files")
    parser.add_argument("--stage", choices=["collect", "train"], required=True, help="Which stage to run.")
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf", help="Model backend for collection.")

    parser.add_argument("--options_path", default=None, help="Path to options.json (list or hierarchical dict)")
    parser.add_argument(
        "--utilities_path",
        default=None,
        help="Path to utilities.json or to a directory of per-role utility files.",
    )
    parser.add_argument(
        "--utilities_dir",
        default=None,
        help="Directory containing per-role utility files (matched by '*_<role_stub>.json').",
    )
    parser.add_argument("--roles", type=none_or_str, default=None, help="Comma-separated roles list")
    parser.add_argument("--roleset", type=none_or_str, default=None, help="Role set key in roles_config_path")
    parser.add_argument("--roles_config_path", type=none_or_str, default=None, help="Path to role_sets.yaml")
    parser.add_argument("--layers", default="all", help="Layer subset spec: all | 0-31 | 0,1,2 | 0-10,12,14-20")
    parser.add_argument(
        "--max_new_tokens_for_parsing",
        type=int,
        default=2,
        help="Generate up to this many tokens to parse rating (activations still taken at first generated token).",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.75,
        help="vLLM gpu_memory_utilization setting for --backend vllm (lower to reduce KV cache allocation).",
    )
    parser.add_argument(
        "--vllm-enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="vLLM: disable CUDA graph capture (default True; use --no-vllm-enforce-eager for max throughput on stable setups).",
    )
    parser.add_argument(
        "--vllm-no-compile",
        action="store_true",
        help="vLLM: pass compilation_config=0 (no torch.compile / vLLM inductor path; slower but fewer GPU driver edge cases).",
    )
    parser.add_argument(
        "--vllm-attention-backend",
        type=str,
        default=None,
        help="vLLM AttentionConfig.backend override, e.g. flash_attn or flashinfer (default: vLLM auto). "
        "Try flash_attn if FlashInfer fails on your stack (requires compatible flash-attn build).",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=1024,
        help="Max sequence length for tokenization (--backend hf) or vLLM KV cache (--backend vllm).",
    )
    parser.add_argument("--progress_every", type=int, default=100, help="Print progress every N examples (0 disables).")
    parser.add_argument(
        "--max_examples",
        type=int,
        default=0,
        help="If > 0, stop after this many prompts (for fast debugging).",
    )
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="Force loading/running on CPU to avoid CUDA errors (slow).",
    )
    parser.add_argument(
        "--hf_device_map_auto",
        action="store_true",
        help="Use Accelerate device_map='auto' (may offload to CPU/disk). Incompatible with --hf_direct_gpu_load.",
    )
    parser.add_argument(
        "--hf_direct_gpu_load",
        action="store_true",
        help="Load shards directly onto cuda:0 (device_map={'': 0}). Faster on native Linux; often breaks on WSL. "
        "Default is CPU-staged load then .to(cuda:0).",
    )
    parser.add_argument(
        "--hf_bnb_8bit",
        action="store_true",
        help="Load model in 8-bit via bitsandbytes on cuda:0 (pip install bitsandbytes). Uses ~8–10 GiB VRAM instead of "
        "~18+ GiB host RAM for CPU staging. Activations differ from full fp16.",
    )
    parser.add_argument(
        "--cuda_launch_blocking",
        action="store_true",
        help="Set CUDA_LAUNCH_BLOCKING=1 to make CUDA stack traces synchronous (slower).",
    )
    parser.add_argument(
        "--attn_implementation",
        type=none_or_str,
        default=None,
        help="Transformers attention implementation override (e.g. eager, sdpa).",
    )
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code for tokenizer/model.")
    parser.add_argument("--fp16", action="store_true", help="Load model in float16 (recommended).")
    parser.add_argument("--bf16", action="store_true", help="Load model in bfloat16.")

    parser.add_argument(
        "--position",
        choices=["prompt_last", "gen_first"],
        default="gen_first",
        help="Which activation position to probe.",
    )
    parser.add_argument("--target", choices=["utility", "rating"], default="utility", help="Regression target.")
    parser.add_argument(
        "--probe_mode",
        choices=["all", "per_role", "cross_role"],
        default="all",
        help="How to split train/test across roles.",
    )
    parser.add_argument("--test_fraction", type=float, default=0.2, help="Fraction of examples for test in random split modes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--ridge_lambda", type=float, default=1.0, help="Ridge regularization strength.")

    args = parser.parse_args()

    if args.cuda_launch_blocking:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    if isinstance(args.save_dir, str):
        args.save_dir = args.save_dir.replace("<model_key>", args.model_key)

    if args.stage == "collect":
        if not args.options_path:
            raise ValueError("--options_path is required for --stage collect")
        if not args.utilities_path and not args.utilities_dir:
            raise ValueError("Provide --utilities_path (file or directory) or --utilities_dir for --stage collect")
        collect(args)
    elif args.stage == "train":
        train(args)
    else:
        raise ValueError(f"Unknown stage: {args.stage}")
