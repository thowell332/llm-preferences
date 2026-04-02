#!/usr/bin/env python3

import argparse
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

# Allow imports from utility_analysis/
sys.path.append("../../")

from compute_utilities.utils import flatten_hierarchical_options  # noqa: E402


RATING_PROMPT_TEMPLATE = """You are a {role}. Consider the following observation about the state of the world. Rate the implied state of the world on a scale of 1-10.

{option}

Please respond with only a number 1-10."""


def none_or_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() == "none":
        return None
    return value


def _load_models_yaml(models_yaml_path: str) -> Dict[str, Any]:
    with open(models_yaml_path, "r") as f:
        return yaml.safe_load(f)


def _resolve_model_paths(models_yaml_path: str, model_key: str) -> Tuple[str, Optional[str]]:
    models_cfg = _load_models_yaml(models_yaml_path)
    if model_key not in models_cfg:
        raise ValueError(f"Unknown model_key '{model_key}' in {models_yaml_path}")

    cfg = models_cfg[model_key]
    # Prefer explicit local path if present (most repo models use this).
    model_path = cfg.get("path") or cfg.get("model_name")
    tokenizer_path = cfg.get("tokenizer_path")

    if not model_path:
        raise ValueError(
            f"Model '{model_key}' has no 'path' or 'model_name' in models.yaml; "
            "cannot load a local Transformers model for activation extraction."
        )
    return model_path, tokenizer_path


def _load_options(options_path: str) -> List[Dict[str, Any]]:
    with open(options_path, "r") as f:
        options_data = json.load(f)
    if isinstance(options_data, dict):
        options_list = flatten_hierarchical_options(options_data)
    elif isinstance(options_data, list):
        options_list = options_data
    else:
        raise ValueError(f"Invalid options type: {type(options_data)}")

    return [{"id": str(i), "description": desc} for i, desc in enumerate(options_list)]


def _load_roles(
    roles: Optional[str],
    roleset: Optional[str],
    roles_config_path: Optional[str],
) -> List[str]:
    if roles:
        return [r.strip() for r in roles.split(",") if r.strip()]
    if not roleset:
        raise ValueError("Must provide either --roles or --roleset")
    if not roles_config_path:
        raise ValueError("When using --roleset you must provide --roles_config_path")
    with open(roles_config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if roleset not in cfg:
        raise ValueError(f"roleset '{roleset}' not found in {roles_config_path}")
    return list(cfg[roleset])


def _load_utilities(utilities_path: str) -> Dict[str, float]:
    with open(utilities_path, "r") as f:
        data = json.load(f)

    # Accept either:
    # - { "<id>": float, ... }
    # - { "utilities": { "<id>": { "mean": float, ... }, ... }, ... } (compute_utilities output)
    # - { "utilities": { "<id>": float, ... }, ... }
    if isinstance(data, dict) and "utilities" in data:
        data = data["utilities"]

    if not isinstance(data, dict):
        raise ValueError("utilities.json must be a dict mapping option_id -> utility (or a compute_utilities results JSON).")

    out: Dict[str, float] = {}
    for k, v in data.items():
        kid = str(k)
        if isinstance(v, (int, float)):
            out[kid] = float(v)
        elif isinstance(v, dict) and "mean" in v:
            out[kid] = float(v["mean"])
        else:
            raise ValueError(f"Unrecognized utility value for id={k!r}: {v!r}")
    return out


def _parse_layers_spec(layers_spec: str, num_layers: int) -> List[int]:
    """
    layers_spec supports:
      - "all"
      - "0-31" (inclusive)
      - "0,1,2,10"
      - "0-10,12,14-20"
    """
    layers_spec = layers_spec.strip().lower()
    if layers_spec == "all":
        return list(range(num_layers))

    parts = [p.strip() for p in layers_spec.split(",") if p.strip()]
    layers: List[int] = []
    for part in parts:
        if "-" in part:
            a, b = part.split("-", 1)
            start = int(a)
            end = int(b)
            if end < start:
                raise ValueError(f"Invalid layer range: {part}")
            layers.extend(list(range(start, end + 1)))
        else:
            layers.append(int(part))

    layers = sorted(set(layers))
    for l in layers:
        if l < 0 or l >= num_layers:
            raise ValueError(f"Layer index out of bounds: {l} (num_layers={num_layers})")
    return layers


def _parse_rating(text: str) -> Optional[int]:
    if text is None:
        return None
    s = text.strip()
    m = re.search(r"\b(10|[1-9])\b", s)
    if not m:
        return None
    val = int(m.group(1))
    if 1 <= val <= 10:
        return val
    return None


@dataclass
class ExampleMeta:
    role: str
    option_id: str
    rating: Optional[int]
    utility: float


@torch.no_grad()
def _get_residual_stream_at_positions(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layers: Sequence[int],
    max_new_tokens_for_parsing: int,
) -> Tuple[str, Optional[int], Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    Returns:
      - decoded_generation (up to max_new_tokens_for_parsing)
      - parsed rating (1..10) or None
      - residuals_prompt_last[layer] -> [hidden_dim] (prompt last token)
      - residuals_gen_first[layer] -> [hidden_dim] (first generated token)
    """
    # We only need a small subset of layers, so avoid `output_hidden_states=True` which
    # forces Transformers to materialize/store hidden states for *all* layers.
    # Instead, capture layer outputs with forward hooks and then slice the token position.
    def _get_transformer_blocks(m: AutoModelForCausalLM) -> Sequence[Any]:
        # LlamaForCausalLM: m.model.layers
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers
        # Some other architectures may use different attribute names; fall back if possible.
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return m.transformer.h
        raise ValueError(
            "Could not locate transformer block modules on the model. "
            "Expected something like model.model.layers (Llama) or model.transformer.h."
        )

    blocks = _get_transformer_blocks(model)
    max_requested_layer = max(layers) if len(layers) > 0 else -1
    if max_requested_layer >= len(blocks):
        raise ValueError(f"Requested layer id {max_requested_layer}, but model has only {len(blocks)} layers.")

    layers_set = list(layers)

    # 1) Forward on prompt to get residual stream at last prompt token.
    prompt_last_index = input_ids.shape[1] - 1
    residuals_prompt_last: Dict[int, torch.Tensor] = {}
    hooks: List[Any] = []

    for l in layers_set:
        def _hook(mod: Any, _inp: Any, out: Any, layer_idx: int = l) -> None:
            # Llama decoder layers often return (hidden_states, ...) but we only need hidden_states.
            hs = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(hs):
                # Keep on-device for now; we'll move to CPU once the forward pass completes.
                residuals_prompt_last[layer_idx] = hs[0, prompt_last_index, :].detach()

        hooks.append(blocks[l].register_forward_hook(_hook))

    out_prompt = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
        # Avoid computing logits for all tokens; we only need the next-token distribution.
        num_logits_to_keep=1,
    )

    for h in hooks:
        h.remove()

    missing = [l for l in layers_set if l not in residuals_prompt_last]
    if missing:
        raise RuntimeError(f"Failed to capture residuals for layers: {missing}")

    # 2) Greedy-generate one token (for activation at first generated token).
    next_token_id = torch.argmax(out_prompt.logits[:, -1, :], dim=-1, keepdim=True)  # [1,1]
    input_ids_1 = torch.cat([input_ids, next_token_id], dim=1)
    attention_mask_1 = torch.cat([attention_mask, torch.ones_like(next_token_id)], dim=1)

    gen_first_index = input_ids_1.shape[1] - 1
    residuals_gen_first: Dict[int, torch.Tensor] = {}
    hooks = []
    for l in layers_set:
        def _hook(mod: Any, _inp: Any, out: Any, layer_idx: int = l) -> None:
            hs = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(hs):
                residuals_gen_first[layer_idx] = hs[0, gen_first_index, :].detach()

        hooks.append(blocks[l].register_forward_hook(_hook))

    out_1 = model(
        input_ids=input_ids_1,
        attention_mask=attention_mask_1,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
        # Only need last-token logits for greedy argmax.
        num_logits_to_keep=1,
    )

    for h in hooks:
        h.remove()

    missing = [l for l in layers_set if l not in residuals_gen_first]
    if missing:
        raise RuntimeError(f"Failed to capture residuals for layers on gen_first: {missing}")

    # 3) Optionally generate a couple more tokens for rating parsing only.
    gen_ids = [int(next_token_id.item())]
    cur_input_ids = input_ids_1
    cur_attention_mask = attention_mask_1
    for _ in range(max_new_tokens_for_parsing - 1):
        out_k = model(
            input_ids=cur_input_ids,
            attention_mask=cur_attention_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
            num_logits_to_keep=1,
        )
        nxt = torch.argmax(out_k.logits[:, -1, :], dim=-1, keepdim=True)
        gen_ids.append(int(nxt.item()))
        cur_input_ids = torch.cat([cur_input_ids, nxt], dim=1)
        cur_attention_mask = torch.cat([cur_attention_mask, torch.ones_like(nxt)], dim=1)

    # Caller will pass tokenizer to decode; keep ids here instead.
    return json.dumps(gen_ids), None, residuals_prompt_last, residuals_gen_first


def _decode_generation(tokenizer: Any, gen_ids_json: str) -> str:
    gen_ids = json.loads(gen_ids_json)
    return tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)


def _extract_hidden_states_from_vllm_output(output: Any) -> Optional[np.ndarray]:
    """Best-effort extraction across vLLM versions."""
    for attr in ("hidden_states", "prompt_hidden_states"):
        val = getattr(output, attr, None)
        if val is not None:
            arr = np.array(val)
            if arr.ndim == 3:
                return arr

    kv = getattr(output, "kv_transfer_params", None)
    if isinstance(kv, dict):
        for key in ("hidden_states", "prompt_hidden_states"):
            if key in kv:
                arr = np.array(kv[key])
                if arr.ndim == 3:
                    return arr
        for key in ("hidden_states_path", "prompt_hidden_states_path", "hidden_state_path"):
            if key in kv and isinstance(kv[key], str) and os.path.exists(kv[key]):
                arr = np.array(np.load(kv[key], allow_pickle=True))
                if arr.ndim == 3:
                    return arr
    return None


def _rankdata(x: np.ndarray) -> np.ndarray:
    """
    Average ranks for ties, 1..n.
    """
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    # tie handling
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            avg = ranks[order[i : j + 1]].mean()
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def _spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    rx = _rankdata(x.astype(np.float64))
    ry = _rankdata(y.astype(np.float64))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx**2).sum()) * np.sqrt((ry**2).sum())
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _ridge_fit_closed_form(X: np.ndarray, y: np.ndarray, ridge_lambda: float) -> Tuple[np.ndarray, float]:
    """
    Fit y ≈ X w + b with ridge on w (not b). Returns (w, b).
    """
    X = X.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    x_mean = X.mean(axis=0, keepdims=True)
    y_mean = y.mean()
    Xc = X - x_mean
    yc = y - y_mean
    d = X.shape[1]
    A = Xc.T @ Xc + ridge_lambda * np.eye(d, dtype=np.float64)
    bvec = Xc.T @ yc
    w = np.linalg.solve(A, bvec)
    b0 = float(y_mean - (x_mean @ w).item())
    return w.astype(np.float32), b0


def _ridge_predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return X @ w + b


def collect(args: argparse.Namespace) -> None:
    os.makedirs(args.save_dir, exist_ok=True)

    models_yaml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "models.yaml")
    models_yaml_path = os.path.abspath(models_yaml_path)

    model_path, tokenizer_path = _resolve_model_paths(models_yaml_path, args.model_key)

    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(tokenizer_path or model_path, trust_remote_code=args.trust_remote_code)
    num_layers = getattr(cfg, "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError("Could not determine num_hidden_layers from model config")
    layers = _parse_layers_spec(args.layers, num_layers)

    options = _load_options(args.options_path)
    roles = _load_roles(args.roles, args.roleset, args.roles_config_path)
    utilities = _load_utilities(args.utilities_path)

    metas: List[ExampleMeta] = []
    prompts: List[str] = []
    for role in roles:
        for opt in options:
            if args.max_examples and len(prompts) >= args.max_examples:
                break
            option_id = str(opt["id"])
            if option_id not in utilities:
                raise ValueError(f"Option id {option_id} missing from utilities.json")
            prompts.append(RATING_PROMPT_TEMPLATE.format(role=role, option=opt["description"]))
            metas.append(ExampleMeta(role=role, option_id=option_id, rating=None, utility=float(utilities[option_id])))
        if args.max_examples and len(prompts) >= args.max_examples:
            break

    prompt_last_list: List[torch.Tensor] = []
    gen_first_list: List[torch.Tensor] = []
    hidden_dim: Optional[int] = None

    if args.backend == "hf":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path, trust_remote_code=args.trust_remote_code)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)

        # Match the previous, simpler behavior: let Accelerate choose an offload plan.
        device_map = None
        if torch.cuda.is_available() and not getattr(args, "force_cpu", False):
            device_map = "auto"

        from_pretrained_kwargs: Dict[str, Any] = {
            "trust_remote_code": args.trust_remote_code,
            "torch_dtype": dtype,
            "device_map": device_map,
            "low_cpu_mem_usage": True,
        }
        if args.attn_implementation:
            from_pretrained_kwargs["attn_implementation"] = args.attn_implementation

        model = AutoModelForCausalLM.from_pretrained(model_path, **from_pretrained_kwargs)
        model.eval()
        print(
            f"[collect:hf] model loaded. device_map={device_map}",
            flush=True,
        )

        total = len(prompts)
        for idx, prompt in enumerate(prompts, start=1):
            # Truncate to avoid blowing up attention + logits on very long option text.
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True,
                max_length=args.max_model_len,
            )
            input_ids = enc["input_ids"].to(model.device)
            attention_mask = enc["attention_mask"].to(model.device)
            if idx == 1:
                print(
                    f"[collect:hf] starting first forward: input_len={int(input_ids.shape[1])} layers={layers}",
                    flush=True,
                )
            gen_ids_json, _, resid_prompt_last, resid_gen_first = _get_residual_stream_at_positions(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                layers=layers,
                max_new_tokens_for_parsing=args.max_new_tokens_for_parsing,
            )
            metas[idx - 1].rating = _parse_rating(_decode_generation(tokenizer, gen_ids_json))
            Xp = torch.stack([resid_prompt_last[l].to(dtype=torch.float16) for l in layers], dim=0)
            Xg = torch.stack([resid_gen_first[l].to(dtype=torch.float16) for l in layers], dim=0)
            if hidden_dim is None:
                hidden_dim = int(Xp.shape[1])
            prompt_last_list.append(Xp.cpu())
            gen_first_list.append(Xg.cpu())
            if args.progress_every and (idx % args.progress_every == 0 or idx == total):
                recent = metas[max(0, idx - args.progress_every) : idx]
                ok = sum(1 for m in recent if m.rating is not None)
                print(f"[collect:hf] {idx}/{total} done (recent parsed ratings: {ok}/{len(recent)})", flush=True)

            if args.max_examples and idx >= args.max_examples:
                print(f"[collect:hf] debug: stopping after max_examples={args.max_examples}", flush=True)
                break
    elif args.backend == "vllm":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path, trust_remote_code=args.trust_remote_code)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        try:
            llm = LLM(
                model=model_path,
                tokenizer=tokenizer_path or model_path,
                trust_remote_code=args.trust_remote_code,
                tensor_parallel_size=max(torch.cuda.device_count(), 1),
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                speculative_config={
                    "method": "extract_hidden_states",
                    "num_speculative_tokens": 1,
                },
            )
        except Exception as e:
            msg = str(e)
            if "extract_hidden_states" in msg or "SpeculativeConfig" in msg or "speculative_config" in msg:
                raise RuntimeError(
                    "Failed to initialize vLLM hidden-state extraction backend. "
                    "Your installed vLLM likely does not support "
                    "speculative_config.method='extract_hidden_states'. "
                    "Upgrade vLLM (e.g. pip install --upgrade 'vllm>=0.18.0'), restart runtime, and rerun. "
                    f"Original error:\n{e}"
                ) from e
            if "Free memory on device" in msg or "gpu_memory_utilization" in msg or "desired GPU memory utilization" in msg:
                raise RuntimeError(
                    "vLLM failed to start due to insufficient GPU memory headroom. "
                    "Try reducing --gpu_memory_utilization (e.g. 0.7 or 0.6) and/or --max_model_len (e.g. 1024/2048), "
                    "and close other GPU processes. "
                    f"Original error:\n{e}"
                ) from e
            raise RuntimeError(
                "Failed to initialize vLLM hidden-state extraction backend. "
                "Original error:\n"
                f"{e}"
            ) from e
        sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens_for_parsing)

        total = len(prompts)
        for idx, prompt in enumerate(prompts, start=1):
            req_outs = llm.generate([prompt], sampling_params)
            if not req_outs:
                raise RuntimeError("vLLM returned no outputs for prompt")
            out = req_outs[0]
            hs = _extract_hidden_states_from_vllm_output(out)
            if hs is None:
                raise RuntimeError(
                    "vLLM hidden states unavailable. This runtime likely has an older vLLM version. "
                    "Upgrade vLLM or use --backend hf with a non-quantized model."
                )
            prompt_token_ids = getattr(out, "prompt_token_ids", None)
            if prompt_token_ids is None:
                prompt_token_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            prompt_len = len(prompt_token_ids)
            prompt_last_idx = prompt_len - 1
            gen_first_idx = prompt_len
            if hs.shape[1] <= gen_first_idx:
                raise RuntimeError(
                    f"Hidden states sequence too short ({hs.shape[1]}) for first generated token index {gen_first_idx}"
                )
            Xp = torch.tensor(np.stack([hs[l, prompt_last_idx, :] for l in layers], axis=0), dtype=torch.float16)
            Xg = torch.tensor(np.stack([hs[l, gen_first_idx, :] for l in layers], axis=0), dtype=torch.float16)
            if hidden_dim is None:
                hidden_dim = int(Xp.shape[1])
            prompt_last_list.append(Xp.cpu())
            gen_first_list.append(Xg.cpu())
            text = out.outputs[0].text if out.outputs else ""
            metas[idx - 1].rating = _parse_rating(text)
            if args.progress_every and (idx % args.progress_every == 0 or idx == total):
                recent = metas[max(0, idx - args.progress_every) : idx]
                ok = sum(1 for m in recent if m.rating is not None)
                print(f"[collect:vllm] {idx}/{total} done (recent parsed ratings: {ok}/{len(recent)})", flush=True)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    if hidden_dim is None:
        raise RuntimeError("No examples collected.")

    # Save metadata and activations.
    save_suffix = args.save_suffix or args.model_key
    out_prefix = os.path.join(args.save_dir, f"linear_probes_{save_suffix}")

    meta_path = out_prefix + "_metadata.jsonl"
    with open(meta_path, "w") as f:
        for m in metas:
            f.write(
                json.dumps(
                    {
                        "role": m.role,
                        "option_id": m.option_id,
                        "rating": m.rating,
                        "utility": m.utility,
                    }
                )
                + "\n"
            )

    layers_path = out_prefix + "_layers.json"
    with open(layers_path, "w") as f:
        json.dump({"layers": layers, "num_layers": num_layers, "hidden_dim": hidden_dim}, f, indent=2)

    X_prompt = torch.stack(prompt_last_list, dim=0)  # [N, L, D]
    X_gen = torch.stack(gen_first_list, dim=0)  # [N, L, D]
    torch.save({"X": X_prompt, "layers": layers, "position": "prompt_last"}, out_prefix + "_X_prompt_last.pt")
    torch.save({"X": X_gen, "layers": layers, "position": "gen_first"}, out_prefix + "_X_gen_first.pt")

    run_meta = {
        "experiment": "linear_probes",
        "prompt_template_version": "v1",
        "backend": args.backend,
        "model_key": args.model_key,
        "model_path": model_path,
        "tokenizer_path": tokenizer_path,
        "options_path": args.options_path,
        "utilities_path": args.utilities_path,
        "roles": roles,
        "layers": layers,
        "max_new_tokens_for_parsing": args.max_new_tokens_for_parsing,
        "gpu_memory_utilization": args.gpu_memory_utilization if args.backend == "vllm" else None,
        "max_model_len": args.max_model_len if args.backend == "vllm" else None,
        "dtype": "fp16" if args.fp16 else ("bf16" if args.bf16 else "fp32"),
    }
    with open(out_prefix + "_run_metadata.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    print(f"[collect] wrote {len(metas)} examples to {meta_path}")
    print(f"[collect] wrote activations to {out_prefix}_X_prompt_last.pt and {out_prefix}_X_gen_first.pt")


def train(args: argparse.Namespace) -> None:
    os.makedirs(args.save_dir, exist_ok=True)
    save_suffix = args.save_suffix or args.model_key
    out_prefix = os.path.join(args.save_dir, f"linear_probes_{save_suffix}")

    meta_path = out_prefix + "_metadata.jsonl"
    layers_path = out_prefix + "_layers.json"
    X_path = out_prefix + ("_X_gen_first.pt" if args.position == "gen_first" else "_X_prompt_last.pt")

    # Load
    metas: List[ExampleMeta] = []
    with open(meta_path, "r") as f:
        for line in f:
            d = json.loads(line)
            metas.append(ExampleMeta(role=d["role"], option_id=str(d["option_id"]), rating=d.get("rating"), utility=float(d["utility"])))

    with open(layers_path, "r") as f:
        layer_info = json.load(f)
    layers: List[int] = list(layer_info["layers"])

    pack = torch.load(X_path, map_location="cpu")
    X = pack["X"]  # [N, L, D]

    # Filter invalid ratings if rating is the target; utilities always exist.
    keep = np.ones(len(metas), dtype=bool)
    if args.target == "rating":
        keep = np.array([m.rating is not None for m in metas], dtype=bool)
    kept_indices = np.where(keep)[0]
    if len(kept_indices) < 10:
        raise ValueError(f"Too few valid examples after filtering: {len(kept_indices)}")

    X = X[kept_indices]
    kept_metas = [metas[i] for i in kept_indices.tolist()]

    y = np.array([m.utility if args.target == "utility" else float(m.rating) for m in kept_metas], dtype=np.float32)
    roles = np.array([m.role for m in kept_metas], dtype=object)

    rng = np.random.RandomState(args.seed)
    n = len(kept_metas)

    def split_indices(idxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        idxs = idxs.copy()
        rng.shuffle(idxs)
        n_test = max(1, int(round(args.test_fraction * len(idxs))))
        test = idxs[:n_test]
        train = idxs[n_test:]
        return train, test

    results: Dict[str, Any] = {
        "experiment": "linear_probes",
        "model_key": args.model_key,
        "save_suffix": save_suffix,
        "position": args.position,
        "target": args.target,
        "ridge_lambda": args.ridge_lambda,
        "test_fraction": args.test_fraction,
        "seed": args.seed,
        "layers": layers,
        "probe_mode": args.probe_mode,
        "metrics_by_layer": {},
    }

    # Helper to run one train/test evaluation per layer.
    def eval_one(train_idx: np.ndarray, test_idx: np.ndarray) -> Dict[int, Dict[str, float]]:
        metrics_by_layer: Dict[int, Dict[str, float]] = {}
        X_np = X.numpy().astype(np.float32, copy=False)  # [N, L, D]
        for li, layer in enumerate(layers):
            Xtr = X_np[train_idx, li, :]
            Xte = X_np[test_idx, li, :]
            ytr = y[train_idx]
            yte = y[test_idx]
            w, b0 = _ridge_fit_closed_form(Xtr, ytr, ridge_lambda=args.ridge_lambda)
            yhat = _ridge_predict(Xte, w, b0)
            mse = float(((yte - yhat) ** 2).mean())
            r2 = _r2_score(yte, yhat)
            spr = _spearmanr(yte, yhat)
            layer_metrics: Dict[str, float] = {"mse": mse, "r2": float(r2), "spearman": float(spr)}
            if args.target == "rating":
                # "Accuracy" for rating target: nearest-integer prediction in [1,10]
                yhat_int = np.clip(np.rint(yhat), 1, 10).astype(np.int32)
                yte_int = np.clip(np.rint(yte), 1, 10).astype(np.int32)
                acc = float((yhat_int == yte_int).mean())
                layer_metrics["accuracy"] = acc
            metrics_by_layer[layer] = layer_metrics
        return metrics_by_layer

    if args.probe_mode == "all":
        train_idx, test_idx = split_indices(np.arange(n))
        results["metrics_by_layer"] = eval_one(train_idx, test_idx)
        results["split"] = {"train_size": int(len(train_idx)), "test_size": int(len(test_idx))}

    elif args.probe_mode == "per_role":
        by_role: Dict[str, Any] = {}
        for role in sorted(set(roles.tolist())):
            role_idxs = np.where(roles == role)[0]
            if len(role_idxs) < 10:
                continue
            train_idx, test_idx = split_indices(role_idxs)
            by_role[role] = {
                "train_size": int(len(train_idx)),
                "test_size": int(len(test_idx)),
                "metrics_by_layer": eval_one(train_idx, test_idx),
            }
        results["by_role"] = by_role

    elif args.probe_mode == "cross_role":
        by_role: Dict[str, Any] = {}
        all_roles = sorted(set(roles.tolist()))
        for test_role in all_roles:
            test_idx = np.where(roles == test_role)[0]
            train_idx = np.where(roles != test_role)[0]
            if len(test_idx) < 10 or len(train_idx) < 10:
                continue
            by_role[test_role] = {
                "train_roles": [r for r in all_roles if r != test_role],
                "train_size": int(len(train_idx)),
                "test_size": int(len(test_idx)),
                "metrics_by_layer": eval_one(train_idx, test_idx),
            }
        results["leave_one_role_out"] = by_role
    else:
        raise ValueError(f"Unknown probe_mode: {args.probe_mode}")

    out_path = out_prefix + f"_probe_results_{args.position}_{args.target}_{args.probe_mode}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[train] wrote probe metrics to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear probes: collect activations and train per-layer probes.")
    parser.add_argument("--model_key", default="llama-31-8b-instruct", help="Model key from models.yaml")
    parser.add_argument("--save_dir", default="results/<model_key>", help="Directory to save outputs")
    parser.add_argument("--save_suffix", type=none_or_str, default=None, help="Custom suffix for saved files")
    parser.add_argument("--stage", choices=["collect", "train"], required=True, help="Which stage to run.")
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf", help="Model backend for collection.")

    # Collect args
    parser.add_argument("--options_path", default=None, help="Path to options.json (list or hierarchical dict)")
    parser.add_argument("--utilities_path", default=None, help="Path to utilities.json (id->float or compute_utilities output)")
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
        "--max_model_len",
        type=int,
        default=1024,
        help="vLLM max_model_len for --backend vllm (controls KV cache size).",
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

    # Train args
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

    # expand <model_key> templating used by experiments.yaml runner
    if isinstance(args.save_dir, str):
        args.save_dir = args.save_dir.replace("<model_key>", args.model_key)

    if args.stage == "collect":
        if not args.options_path or not args.utilities_path:
            raise ValueError("--options_path and --utilities_path are required for --stage collect")
        collect(args)
    elif args.stage == "train":
        train(args)
    else:
        raise ValueError(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    main()

