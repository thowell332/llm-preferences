from __future__ import annotations

import argparse
import inspect
import traceback
from typing import Any, Dict, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from lp.debug import debug_rss, warn_if_risky_model_path


def build_hf_from_pretrained_kwargs(
    args: argparse.Namespace,
    dtype: torch.dtype,
    model_path: str,
) -> Tuple[Dict[str, Any], bool]:
    """
    Returns (kwargs for AutoModelForCausalLM.from_pretrained, move_to_cuda_after_load).
    """
    base_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=args.trust_remote_code)
    bundled_quant = getattr(base_cfg, "quantization_config", None) is not None

    use_bnb_requested = (
        bool(getattr(args, "hf_bnb_8bit", False))
        and torch.cuda.is_available()
        and not getattr(args, "force_cpu", False)
    )
    # BitsAndBytes must not be merged with checkpoints that already declare another scheme
    # (e.g. CompressedTensors, GPTQ export) — Transformers raises ValueError if both are passed.
    use_bnb_8bit = use_bnb_requested and not bundled_quant
    if use_bnb_requested and bundled_quant:
        print(
            "[collect:hf] NOTE: checkpoint already has quantization_config "
            f"({type(base_cfg.quantization_config).__name__}); skipping BitsAndBytes and loading bundled weights.",
            flush=True,
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
    elif bundled_quant and torch.cuda.is_available():
        # device_map onto GPU still goes through meta init + _load_state_dict_into_meta_model, which breaks
        # compressed-tensors / non-float shards on several torch+transformers versions. Load on CPU then .to(cuda).
        device_map = None
        move_to_cuda_after_load = True
        print(
            "[collect:hf] Bundled quantized checkpoint: loading on CPU (device_map=None) then moving to cuda:0 "
            "to avoid meta-model shard load.",
            flush=True,
        )
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

    # Meta init + shard load (`low_cpu_mem_usage=True`) can break bundled quantized checkpoints
    # (compressed-tensors, etc.): weights are non-float and PyTorch rejects requires_grad on load.
    low_mem = not bundled_quant
    if bundled_quant:
        print(
            "[collect:hf] Bundled quantized checkpoint: using low_cpu_mem_usage=False for reliable load "
            "(higher peak RAM than meta init).",
            flush=True,
        )

    from_pretrained_kwargs: Dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "device_map": device_map,
        "low_cpu_mem_usage": low_mem,
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
    elif not bundled_quant:
        try:
            _sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
            if "dtype" in _sig.parameters:
                from_pretrained_kwargs["dtype"] = dtype
            else:
                from_pretrained_kwargs["torch_dtype"] = dtype
        except Exception:
            from_pretrained_kwargs["dtype"] = dtype
    else:
        print(
            "[collect:hf] Bundled quantized checkpoint: omitting extra dtype override; using config on disk.",
            flush=True,
        )
    if args.attn_implementation:
        from_pretrained_kwargs["attn_implementation"] = args.attn_implementation

    if move_to_cuda_after_load:
        if bundled_quant:
            print(
                "[collect:hf] NOTE: Quantized checkpoint loads weights on CPU first; ensure enough RAM. "
                "If load or .to(cuda) fails, use --backend vllm for this model.",
                flush=True,
            )
        else:
            print(
                "[collect:hf] NOTE: CPU-staged load needs ~18+ GiB free RAM for 8B fp16. "
                "Exit -9 during shard load = OOM killer (raise WSL .wslconfig memory, free RAM, copy model off OneDrive, "
                "or use --hf_bnb_8bit).",
                flush=True,
            )
    warn_if_risky_model_path(model_path)

    print(
        f"[collect:hf] calling AutoModelForCausalLM.from_pretrained (this may take several minutes)… "
        f"device_map={device_map!r} move_to_cuda_after_load={move_to_cuda_after_load} "
        f"hf_bnb_8bit={use_bnb_8bit} bundled_quant={bundled_quant}",
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
