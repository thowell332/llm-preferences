from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from lp.activations import (
    decode_generation,
    hidden_states_from_vllm_output,
    residual_stream_at_positions,
)
from lp.data import (
    ExampleMeta,
    RATING_PROMPT_TEMPLATE,
    load_options,
    load_roles,
    load_utilities,
    models_yaml_path_for_experiment,
    parse_layers_spec,
    parse_rating,
    resolve_model_paths,
)
from lp.debug import print_collect_startup
from lp.hf_loader import build_hf_from_pretrained_kwargs, finalize_hf_model_on_device, load_hf_causal_lm


def _role_stub(role: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", role).strip("_").lower() or "role"


def _load_role_utilities_from_dir(utilities_dir: Path, role: str) -> Dict[str, float]:
    role_stub = _role_stub(role)
    suffix = f"_{role_stub}.json"
    candidates = sorted(p for p in utilities_dir.glob(f"results_utilities*{suffix}") if p.is_file())
    if not candidates:
        raise FileNotFoundError(
            f"No utility file found for role {role!r} in {utilities_dir}. "
            f"Expected a filename matching 'results_utilities*{suffix}'."
        )
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise ValueError(
            f"Multiple utility files match role {role!r} in {utilities_dir}: {names}. "
            f"Expected exactly one filename matching 'results_utilities*{suffix}'."
        )
    return load_utilities(str(candidates[0]))


def _resolve_role_to_utilities(args: argparse.Namespace, roles: List[str]) -> Dict[str, Dict[str, float]]:
    raw_dir = getattr(args, "utilities_dir", None)
    raw_path = getattr(args, "utilities_path", None)

    utilities_dir: Optional[Path] = None
    if raw_dir:
        utilities_dir = Path(str(raw_dir)).expanduser().resolve()
    elif raw_path and Path(str(raw_path)).expanduser().is_dir():
        utilities_dir = Path(str(raw_path)).expanduser().resolve()

    if utilities_dir is not None:
        if not utilities_dir.is_dir():
            raise NotADirectoryError(f"utilities_dir is not a directory: {utilities_dir}")
        return {role: _load_role_utilities_from_dir(utilities_dir, role) for role in roles}

    if not raw_path:
        raise ValueError("Missing utility source: provide --utilities_path or --utilities_dir")
    shared = load_utilities(str(raw_path))
    return {role: shared for role in roles}


def _vllm_draft_model_config_for_extract_hidden_states(
    model_path: str,
    tokenizer_path: Optional[str],
    trust_remote_code: bool,
) -> Any:
    """
    vLLM V1 speculative method ``extract_hidden_states`` requires
    ``eagle_aux_hidden_state_layer_ids`` on the draft HF config. That is copied
    from SpeculativeConfig.draft_model_config.hf_config.to_dict() before vLLM
    wraps it in ExtractHiddenStatesConfig; passing only method/num_spec_tokens
    leaves it empty and engine startup fails.

    We inject the layer list via ModelConfig.hf_overrides so it lands on the
    loaded Llama (etc.) config. Use all transformer layers so downstream code
    can index hidden states by global layer id.
    """
    from vllm.config import ModelConfig

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    n_layers = int(getattr(cfg, "num_hidden_layers"))
    aux_ids = list(range(n_layers))
    print(
        f"[collect:vllm] extract_hidden_states: eagle_aux_hidden_state_layer_ids "
        f"= 0..{n_layers - 1} ({n_layers} layers)",
        flush=True,
    )
    return ModelConfig(
        model=model_path,
        tokenizer=tokenizer_path or model_path,
        trust_remote_code=trust_remote_code,
        hf_overrides={"eagle_aux_hidden_state_layer_ids": aux_ids},
    )


def collect_hf(
    args: argparse.Namespace,
    model_path: str,
    tokenizer_path: Optional[str],
    layers: List[int],
    prompts: List[str],
    metas: List[ExampleMeta],
) -> tuple[List[torch.Tensor], List[torch.Tensor], Optional[int]]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)

    fp_kwargs, move_to_cuda_after_load = build_hf_from_pretrained_kwargs(args, dtype, model_path)
    model = load_hf_causal_lm(model_path, fp_kwargs)
    model = finalize_hf_model_on_device(model, move_to_cuda_after_load)
    print(
        f"[collect:hf] model ready. device_map={fp_kwargs.get('device_map')!r} model.device={getattr(model, 'device', None)}",
        flush=True,
    )

    prompt_last_list: List[torch.Tensor] = []
    gen_first_list: List[torch.Tensor] = []
    hidden_dim: Optional[int] = None
    total = len(prompts)

    for idx, prompt in enumerate(prompts, start=1):
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
        gen_ids_json, _, resid_prompt_last, resid_gen_first = residual_stream_at_positions(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            layers=layers,
            max_new_tokens_for_parsing=args.max_new_tokens_for_parsing,
        )
        metas[idx - 1].rating = parse_rating(decode_generation(tokenizer, gen_ids_json))
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

    return prompt_last_list, gen_first_list, hidden_dim


def _vllm_attention_config_from_args(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    raw = getattr(args, "vllm_attention_backend", None)
    if not raw or str(raw).strip().lower() in ("", "auto", "none"):
        return None
    key = str(raw).strip().replace("-", "_").upper()
    return {"backend": key}


def collect_vllm(
    args: argparse.Namespace,
    model_path: str,
    tokenizer_path: Optional[str],
    layers: List[int],
    prompts: List[str],
    metas: List[ExampleMeta],
) -> tuple[List[torch.Tensor], List[torch.Tensor], Optional[int]]:
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    draft_model_config = _vllm_draft_model_config_for_extract_hidden_states(
        model_path, tokenizer_path, args.trust_remote_code
    )

    attention_config = _vllm_attention_config_from_args(args)
    enforce_eager = bool(getattr(args, "vllm_enforce_eager", True))
    compilation_config = 0 if bool(getattr(args, "vllm_no_compile", False)) else None
    print(
        f"[collect:vllm] LLM kwargs: enforce_eager={enforce_eager} "
        f"compilation_config={compilation_config!r} attention_config={attention_config!r}",
        flush=True,
    )

    with tempfile.TemporaryDirectory() as hidden_states_storage:
        try:
            llm_kwargs: Dict[str, Any] = dict(
                model=model_path,
                tokenizer=tokenizer_path or model_path,
                trust_remote_code=args.trust_remote_code,
                tensor_parallel_size=max(torch.cuda.device_count(), 1),
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                enforce_eager=enforce_eager,
                speculative_config={
                    "method": "extract_hidden_states",
                    "num_speculative_tokens": 1,
                    "draft_model_config": draft_model_config,
                },
                kv_transfer_config={
                    "kv_connector": "ExampleHiddenStatesConnector",
                    "kv_role": "kv_producer",
                    "kv_connector_extra_config": {"shared_storage_path": hidden_states_storage},
                },
            )
            if attention_config is not None:
                llm_kwargs["attention_config"] = attention_config
            if compilation_config is not None:
                llm_kwargs["compilation_config"] = compilation_config
            llm = LLM(**llm_kwargs)
        except Exception as e:
            msg = str(e)
            if "eagle_aux_hidden_state_layer_ids" in msg:
                raise RuntimeError(
                    "vLLM extract_hidden_states failed: draft config missing "
                    "eagle_aux_hidden_state_layer_ids. This is unexpected after "
                    "linear_probes injects it via ModelConfig.hf_overrides; try "
                    "upgrading vLLM or report the stack trace. "
                    f"Original error:\n{e}"
                ) from e
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
        prompt_last_list: List[torch.Tensor] = []
        gen_first_list: List[torch.Tensor] = []
        hidden_dim: Optional[int] = None
        total = len(prompts)

        for idx, prompt in enumerate(prompts, start=1):
            try:
                req_outs = llm.generate([prompt], sampling_params)
            except Exception as e:
                msg = str(e)
                is_engine_dead = type(e).__name__ == "EngineDeadError"
                is_cuda_msg = (
                    "CUDA" in msg
                    or "cuda" in msg
                    or "AcceleratorError" in type(e).__name__
                    or "CUDA error" in msg
                )
                if is_engine_dead or is_cuda_msg:
                    raise RuntimeError(
                        "vLLM failed during generation (often WSL2 / driver / FlashInfer). "
                        "Try: rerun with default --vllm-enforce-eager (on), add --vllm-no-compile, "
                        "lower --gpu_memory_utilization, use --cuda_launch_blocking for a clearer stack, "
                        "or --vllm-attention-backend flash_attn if flash-attn is installed. "
                        "If the GPU was left in a bad state, reboot WSL (`wsl --shutdown`) or the host. "
                        f"Original error:\n{e}"
                    ) from e
                raise
            if not req_outs:
                raise RuntimeError("vLLM returned no outputs for prompt")
            out = req_outs[0]
            hs = hidden_states_from_vllm_output(out)
            if hs is None:
                raise RuntimeError(
                    "vLLM hidden states unavailable after generate(). "
                    "Expected output.kv_transfer_params['hidden_states_path'] from "
                    "ExampleHiddenStatesConnector (vLLM >= 0.18). Check vLLM logs or try upgrading vLLM."
                )
            prompt_token_ids = getattr(out, "prompt_token_ids", None)
            if prompt_token_ids is None:
                prompt_token_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            prompt_token_ids = [int(x) for x in prompt_token_ids]
            prompt_len = len(prompt_token_ids)
            prompt_last_idx = prompt_len - 1
            gen_first_idx = prompt_len
            if hs.shape[1] <= prompt_last_idx:
                raise RuntimeError(
                    f"Hidden states sequence too short ({hs.shape[1]}) for prompt_last index {prompt_last_idx}"
                )
            Xp = torch.tensor(np.stack([hs[l, prompt_last_idx, :] for l in layers], axis=0), dtype=torch.float16)

            comp0 = out.outputs[0] if out.outputs else None
            sampled = list(getattr(comp0, "token_ids", None) or []) if comp0 is not None else []
            if not sampled:
                raise RuntimeError(
                    "vLLM returned no completion token_ids; need the first sampled token to run a "
                    "second prefill for gen-first hidden states (ExampleHiddenStatesConnector only records the prompt)."
                )
            extended_ids = prompt_token_ids + [int(sampled[0])]
            # vLLM rejects max_tokens=0 (VLLMValidationError); one decode step is fine—we only read prefill HS.
            sp_prefill = SamplingParams(temperature=0.0, max_tokens=1)
            out2_list = llm.generate([{"prompt_token_ids": extended_ids}], sp_prefill)
            if not out2_list:
                raise RuntimeError("vLLM returned no outputs for extended-token prefill")
            out2 = out2_list[0]
            hs2 = hidden_states_from_vllm_output(out2)
            if hs2 is None:
                raise RuntimeError(
                    "vLLM hidden states unavailable after extended-prompt generate(); "
                    "check kv_transfer_params / connector logs."
                )
            if hs2.shape[1] <= gen_first_idx:
                raise RuntimeError(
                    f"Second-pass hidden states too short ({hs2.shape[1]}) for first-generated index {gen_first_idx} "
                    f"(expected prefill length >= {prompt_len + 1})."
                )
            Xg = torch.tensor(np.stack([hs2[l, gen_first_idx, :] for l in layers], axis=0), dtype=torch.float16)
            if hidden_dim is None:
                hidden_dim = int(Xp.shape[1])
            prompt_last_list.append(Xp.cpu())
            gen_first_list.append(Xg.cpu())
            text = out.outputs[0].text if out.outputs else ""
            metas[idx - 1].rating = parse_rating(text)
            if args.progress_every and (idx % args.progress_every == 0 or idx == total):
                recent = metas[max(0, idx - args.progress_every) : idx]
                ok = sum(1 for m in recent if m.rating is not None)
                print(f"[collect:vllm] {idx}/{total} done (recent parsed ratings: {ok}/{len(recent)})", flush=True)

        return prompt_last_list, gen_first_list, hidden_dim


def collect(args: argparse.Namespace) -> None:
    os.makedirs(args.save_dir, exist_ok=True)

    model_path, tokenizer_path = resolve_model_paths(models_yaml_path_for_experiment(), args.model_key)

    cfg = AutoConfig.from_pretrained(tokenizer_path or model_path, trust_remote_code=args.trust_remote_code)
    num_layers = getattr(cfg, "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError("Could not determine num_hidden_layers from model config")
    layers = parse_layers_spec(args.layers, num_layers)

    options = load_options(args.options_path)
    roles = load_roles(args.roles, args.roleset, args.roles_config_path)
    role_to_utilities = _resolve_role_to_utilities(args, roles)

    metas: List[ExampleMeta] = []
    prompts: List[str] = []
    for role in roles:
        utilities = role_to_utilities[role]
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

    print_collect_startup(args, model_path, tokenizer_path, num_layers)

    if args.backend == "hf":
        prompt_last_list, gen_first_list, hidden_dim = collect_hf(
            args, model_path, tokenizer_path, layers, prompts, metas
        )
    elif args.backend == "vllm":
        prompt_last_list, gen_first_list, hidden_dim = collect_vllm(
            args, model_path, tokenizer_path, layers, prompts, metas
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    if hidden_dim is None:
        raise RuntimeError("No examples collected.")

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

    X_prompt = torch.stack(prompt_last_list, dim=0)
    X_gen = torch.stack(gen_first_list, dim=0)
    torch.save({"X": X_prompt, "layers": layers, "position": "prompt_last"}, out_prefix + "_X_prompt_last.pt")
    torch.save({"X": X_gen, "layers": layers, "position": "gen_first"}, out_prefix + "_X_gen_first.pt")

    run_meta: Dict[str, Any] = {
        "experiment": "linear_probes",
        "prompt_template_version": "v1",
        "backend": args.backend,
        "model_key": args.model_key,
        "model_path": model_path,
        "tokenizer_path": tokenizer_path,
        "options_path": args.options_path,
        "utilities_path": args.utilities_path,
        "utilities_dir": getattr(args, "utilities_dir", None),
        "roles": roles,
        "layers": layers,
        "max_new_tokens_for_parsing": args.max_new_tokens_for_parsing,
        "gpu_memory_utilization": args.gpu_memory_utilization if args.backend == "vllm" else None,
        "max_model_len": args.max_model_len if args.backend == "vllm" else None,
        "vllm_enforce_eager": bool(getattr(args, "vllm_enforce_eager", True)) if args.backend == "vllm" else None,
        "vllm_no_compile": bool(getattr(args, "vllm_no_compile", False)) if args.backend == "vllm" else None,
        "vllm_attention_backend": getattr(args, "vllm_attention_backend", None) if args.backend == "vllm" else None,
        "dtype": "fp16" if args.fp16 else ("bf16" if args.bf16 else "fp32"),
        "hf_bnb_8bit": bool(getattr(args, "hf_bnb_8bit", False)) if args.backend == "hf" else None,
    }
    with open(out_prefix + "_run_metadata.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    print(f"[collect] wrote {len(metas)} examples to {meta_path}")
    print(f"[collect] wrote activations to {out_prefix}_X_prompt_last.pt and {out_prefix}_X_gen_first.pt")
