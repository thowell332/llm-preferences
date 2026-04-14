from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM


@torch.no_grad()
def residual_stream_at_positions(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layers: Sequence[int],
    max_new_tokens_for_parsing: int,
) -> Tuple[str, Optional[int], Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    Returns:
      - JSON list of generated token ids (for decoding)
      - placeholder None (rating parsed by caller after decode)
      - residuals_prompt_last[layer] -> [hidden_dim]
      - residuals_gen_first[layer] -> [hidden_dim]
    """

    def transformer_blocks(m: AutoModelForCausalLM) -> Sequence[Any]:
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return m.transformer.h
        raise ValueError(
            "Could not locate transformer block modules on the model. "
            "Expected something like model.model.layers (Llama) or model.transformer.h."
        )

    blocks = transformer_blocks(model)
    max_requested_layer = max(layers) if len(layers) > 0 else -1
    if max_requested_layer >= len(blocks):
        raise ValueError(f"Requested layer id {max_requested_layer}, but model has only {len(blocks)} layers.")

    layers_set = list(layers)
    prompt_last_index = input_ids.shape[1] - 1
    residuals_prompt_last: Dict[int, torch.Tensor] = {}
    hooks: List[Any] = []

    for l in layers_set:
        def _hook(mod: Any, _inp: Any, out: Any, layer_idx: int = l) -> None:
            hs = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(hs):
                residuals_prompt_last[layer_idx] = hs[0, prompt_last_index, :].detach()

        hooks.append(blocks[l].register_forward_hook(_hook))

    out_prompt = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
        num_logits_to_keep=1,
    )

    for h in hooks:
        h.remove()

    missing = [l for l in layers_set if l not in residuals_prompt_last]
    if missing:
        raise RuntimeError(f"Failed to capture residuals for layers: {missing}")

    next_token_id = torch.argmax(out_prompt.logits[:, -1, :], dim=-1, keepdim=True)
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
        num_logits_to_keep=1,
    )

    for h in hooks:
        h.remove()

    missing = [l for l in layers_set if l not in residuals_gen_first]
    if missing:
        raise RuntimeError(f"Failed to capture residuals for layers on gen_first: {missing}")

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

    return json.dumps(gen_ids), None, residuals_prompt_last, residuals_gen_first


@torch.no_grad()
def residual_stream_at_prompt_positions(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layers: Sequence[int],
    positions: Sequence[int],
) -> Dict[int, Dict[int, torch.Tensor]]:
    """
    Capture residual stream vectors at specified prompt token positions.

    Returns:
      position_to_layer_to_vec[position][layer] -> [hidden_dim]
    """

    def transformer_blocks(m: AutoModelForCausalLM) -> Sequence[Any]:
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return m.transformer.h
        raise ValueError(
            "Could not locate transformer block modules on the model. "
            "Expected something like model.model.layers (Llama) or model.transformer.h."
        )

    blocks = transformer_blocks(model)
    layers_set = list(layers)
    pos_list = [int(p) for p in positions]
    seq_len = int(input_ids.shape[1])
    if any(p < 0 or p >= seq_len for p in pos_list):
        raise ValueError(f"Prompt position out of bounds for seq_len={seq_len}: {pos_list}")
    max_requested_layer = max(layers_set) if len(layers_set) > 0 else -1
    if max_requested_layer >= len(blocks):
        raise ValueError(f"Requested layer id {max_requested_layer}, but model has only {len(blocks)} layers.")

    out: Dict[int, Dict[int, torch.Tensor]] = {p: {} for p in pos_list}
    hooks: List[Any] = []
    for l in layers_set:
        def _hook(mod: Any, _inp: Any, layer_out: Any, layer_idx: int = l) -> None:
            hs = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out
            if torch.is_tensor(hs):
                for p in pos_list:
                    out[p][layer_idx] = hs[0, p, :].detach()

        hooks.append(blocks[l].register_forward_hook(_hook))

    _ = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
        num_logits_to_keep=1,
    )

    for h in hooks:
        h.remove()

    for p in pos_list:
        missing = [l for l in layers_set if l not in out[p]]
        if missing:
            raise RuntimeError(f"Failed to capture residuals for positions/layers: pos={p} missing_layers={missing}")
    return out


def decode_generation(tokenizer: Any, gen_ids_json: str) -> str:
    gen_ids = json.loads(gen_ids_json)
    return tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)


def _transpose_slh_to_lsh(arr: np.ndarray, n_tokens: Optional[int]) -> np.ndarray:
    """
    vLLM's extract_hidden_states + ExampleHiddenStatesConnector stores activations as
    [num_tokens, num_hidden_layers, hidden_size] (see ExtractHiddenStatesModel.forward).
    Downstream code indexes [layer, position, :].
    """
    if n_tokens is None or arr.ndim != 3:
        return arr
    if arr.shape[0] == n_tokens and arr.shape[1] < n_tokens:
        return np.ascontiguousarray(np.transpose(arr, (1, 0, 2)))
    return arr


def _hidden_states_array_from_file(path: str) -> Optional[np.ndarray]:
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as f:
            if "hidden_states" not in f.keys():
                return None
            t = f.get_tensor("hidden_states")
            arr = t.detach().float().numpy()
            if arr.ndim != 3:
                return None
            n_tok: Optional[int] = None
            if "token_ids" in f.keys():
                tid = f.get_tensor("token_ids")
                n_tok = int(tid.numel())
            return _transpose_slh_to_lsh(arr, n_tok)
    except Exception:
        pass
    try:
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.ndarray) and arr.ndim == 3:
            return arr
    except Exception:
        pass
    return None


def hidden_states_from_vllm_output(output: Any) -> Optional[np.ndarray]:
    n_tok: Optional[int] = None
    pt = getattr(output, "prompt_token_ids", None)
    if pt is not None:
        n_tok = len(pt)

    for attr in ("hidden_states", "prompt_hidden_states"):
        val = getattr(output, attr, None)
        if val is not None:
            arr = np.array(val)
            if arr.ndim == 3:
                return _transpose_slh_to_lsh(arr, n_tok)

    kv = getattr(output, "kv_transfer_params", None)
    if isinstance(kv, dict):
        for key in ("hidden_states", "prompt_hidden_states"):
            if key in kv:
                arr = np.array(kv[key])
                if arr.ndim == 3:
                    return _transpose_slh_to_lsh(arr, n_tok)
        for key in ("hidden_states_path", "prompt_hidden_states_path", "hidden_state_path"):
            if key in kv and isinstance(kv[key], str) and os.path.exists(kv[key]):
                arr = _hidden_states_array_from_file(kv[key])
                if arr is not None and arr.ndim == 3:
                    return arr
    return None
