from __future__ import annotations

import argparse
import os
from typing import Optional

import torch


def warn_if_risky_model_path(model_path: str) -> None:
    norm = os.path.normpath(model_path)
    if "OneDrive" in norm or norm.startswith("/mnt/"):
        print(
            f"[collect:hf] WARNING: model_path is on a Windows or /mnt mount ({model_path!r}). "
            "Copy the snapshot to Linux local storage (e.g. ~/hf_models/...) when possible.",
            flush=True,
        )


def debug_rss(tag: str) -> None:
    try:
        import psutil  # type: ignore

        rss_gb = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
        print(f"[collect:debug] {tag}: RSS ~{rss_gb:.2f} GiB", flush=True)
    except Exception:
        pass


def print_collect_startup(
    args: argparse.Namespace,
    model_path: str,
    tokenizer_path: Optional[str],
    num_layers: int,
) -> None:
    print("[collect:debug] ========== linear_probes collect ==========", flush=True)
    print(f"[collect:debug] pid={os.getpid()} cwd={os.getcwd()}", flush=True)
    print(f"[collect:debug] torch={getattr(torch, '__version__', '?')} cuda_available={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[collect:debug] cuda_device_count={torch.cuda.device_count()}", flush=True)
    print(f"[collect:debug] model_path={model_path}", flush=True)
    print(f"[collect:debug] tokenizer_path={tokenizer_path or model_path}", flush=True)
    print(f"[collect:debug] num_hidden_layers={num_layers} backend={args.backend}", flush=True)
    print(
        f"[collect:debug] fp16={args.fp16} bf16={args.bf16} force_cpu={getattr(args, 'force_cpu', False)} "
        f"attn_implementation={getattr(args, 'attn_implementation', None)!r}",
        flush=True,
    )
    debug_rss("startup")
    if getattr(args, "force_cpu", False):
        print(
            "[collect:debug] NOTE: --force_cpu loads and runs on CPU (CUDA may still be present but unused). "
            "An 8B fp16 checkpoint needs ~16 GiB weights RAM plus overhead; if the process dies mid-shard "
            "with no Python traceback, suspect OOM (OOM killer / WSL memory cap). Remove --force_cpu to use GPU.",
            flush=True,
        )
