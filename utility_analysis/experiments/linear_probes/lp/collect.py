from __future__ import annotations

import argparse
import json
import os
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
    prompt_last_list: List[torch.Tensor] = []
    gen_first_list: List[torch.Tensor] = []
    hidden_dim: Optional[int] = None
    total = len(prompts)

    for idx, prompt in enumerate(prompts, start=1):
        req_outs = llm.generate([prompt], sampling_params)
        if not req_outs:
            raise RuntimeError("vLLM returned no outputs for prompt")
        out = req_outs[0]
        hs = hidden_states_from_vllm_output(out)
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
    utilities = load_utilities(args.utilities_path)

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
        "roles": roles,
        "layers": layers,
        "max_new_tokens_for_parsing": args.max_new_tokens_for_parsing,
        "gpu_memory_utilization": args.gpu_memory_utilization if args.backend == "vllm" else None,
        "max_model_len": args.max_model_len if args.backend == "vllm" else None,
        "dtype": "fp16" if args.fp16 else ("bf16" if args.bf16 else "fp32"),
        "hf_bnb_8bit": bool(getattr(args, "hf_bnb_8bit", False)) if args.backend == "hf" else None,
    }
    with open(out_prefix + "_run_metadata.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    print(f"[collect] wrote {len(metas)} examples to {meta_path}")
    print(f"[collect] wrote activations to {out_prefix}_X_prompt_last.pt and {out_prefix}_X_gen_first.pt")
