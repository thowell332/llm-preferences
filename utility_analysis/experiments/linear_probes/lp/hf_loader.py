from __future__ import annotations

import argparse
import inspect
import traceback
from typing import Any, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM

from lp.debug import debug_rss, warn_if_risky_model_path


def build_hf_from_pretrained_kwargs(
    args: argparse.Namespace,
    dtype: torch.dtype,
    model_path: str,
) -> Tuple[Dict[str, Any], bool]:
    """
    Returns (kwargs for AutoModelForCausalLM.from_pretrained, move_to_cuda_after_load).
    """
    use_bnb_8bit = (
        bool(getattr(args, "hf_bnb_8bit", False))
        and torch.cuda.is_available()
        and not getattr(args, "force_cpu", False)
    )
    if use_bnb_8bit and (
        getattr(args, "hf_device_map_auto", False) or getattr(args, "hf_direct_gpu_load", False)
    ):
        print("[collect:hf] NOTE: --hf_bnb_8bit takes precedence over --hf_device_map_auto / --hf_direct_gpu_load.", flush=True)

    device_map: Any = None
    move_to_cuda_after_load = False
    if getattr(args, "force_cpu", False):
        device_map = None
    elif use_bnb_8bit:
        device_map = {"": 0}
        move_to_cuda_after_load = False
    elif torch.cuda.is_available():
        if getattr(args, "hf_device_map_auto", False):
            print(
                "[collect:hf] WARNING: --hf_device_map_auto uses device_map='auto' (CPU/disk offload possible; "
                "forward may fail on some setups).",
                flush=True,
            )
            device_map = "auto"
        elif getattr(args, "hf_direct_gpu_load", False):
            device_map = {"": 0}
        else:
            device_map = None
            move_to_cuda_after_load = True
    else:
        device_map = None

    from_pretrained_kwargs: Dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
    }
    if use_bnb_8bit:
        try:
            from transformers import BitsAndBytesConfig  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Install bitsandbytes (pip install bitsandbytes) to use --hf_bnb_8bit."
            ) from e
        from_pretrained_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        print(
            "[collect:hf] Using bitsandbytes 8-bit GPU load (lower VRAM; activations are not full fp16).",
            flush=True,
        )
    else:
        try:
            _sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
            if "dtype" in _sig.parameters:
                from_pretrained_kwargs["dtype"] = dtype
            else:
                from_pretrained_kwargs["torch_dtype"] = dtype
        except Exception:
            from_pretrained_kwargs["dtype"] = dtype
    if args.attn_implementation:
        from_pretrained_kwargs["attn_implementation"] = args.attn_implementation

    if move_to_cuda_after_load:
        print(
            "[collect:hf] NOTE: CPU-staged load needs ~18+ GiB free RAM for 8B fp16. "
            "Exit -9 during shard load = OOM killer (raise WSL .wslconfig memory, free RAM, copy model off OneDrive, "
            "or use --hf_bnb_8bit).",
            flush=True,
        )
    warn_if_risky_model_path(model_path)

    use_bnb_flag = use_bnb_8bit
    print(
        f"[collect:hf] calling AutoModelForCausalLM.from_pretrained (this may take several minutes)… "
        f"device_map={device_map!r} move_to_cuda_after_load={move_to_cuda_after_load} hf_bnb_8bit={use_bnb_flag}",
        flush=True,
    )
    return from_pretrained_kwargs, move_to_cuda_after_load


def load_hf_causal_lm(model_path: str, from_pretrained_kwargs: Dict[str, Any]) -> AutoModelForCausalLM:
    debug_rss("before from_pretrained")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **from_pretrained_kwargs)
    except Exception as e:
        print(f"[collect:hf] from_pretrained FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        raise
    debug_rss("after from_pretrained")
    return model


def finalize_hf_model_on_device(model: AutoModelForCausalLM, move_to_cuda_after_load: bool) -> AutoModelForCausalLM:
    model.eval()
    if move_to_cuda_after_load:
        print("[collect:hf] moving model to cuda:0 after CPU-staged load…", flush=True)
        try:
            model = model.to("cuda:0")
        except Exception as e:
            print(f"[collect:hf] model.to(cuda:0) FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            raise
        debug_rss("after model.to(cuda:0)")
    return model
