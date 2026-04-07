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


def decode_generation(tokenizer: Any, gen_ids_json: str) -> str:
    gen_ids = json.loads(gen_ids_json)
    return tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)


def hidden_states_from_vllm_output(output: Any) -> Optional[np.ndarray]:
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
