from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml

from compute_utilities.utils import flatten_hierarchical_options

RATING_PROMPT_TEMPLATE = """You are a {role}. Consider the following observation about the state of the world. Rate the implied state of the world on a scale of 1-10.

{option}

Please respond with only a number 1-10."""


def none_or_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() == "none":
        return None
    return value


def load_models_yaml(models_yaml_path: str) -> Dict[str, Any]:
    with open(models_yaml_path, "r") as f:
        return yaml.safe_load(f)


def resolve_model_paths(models_yaml_path: str, model_key: str) -> Tuple[str, Optional[str]]:
    models_cfg = load_models_yaml(models_yaml_path)
    if model_key not in models_cfg:
        raise ValueError(f"Unknown model_key '{model_key}' in {models_yaml_path}")

    cfg = models_cfg[model_key]
    model_path = cfg.get("path") or cfg.get("model_name")
    tokenizer_path = cfg.get("tokenizer_path")

    if not model_path:
        raise ValueError(
            f"Model '{model_key}' has no 'path' or 'model_name' in models.yaml; "
            "cannot load a local Transformers model for activation extraction."
        )
    return model_path, tokenizer_path


def load_options(options_path: str) -> List[Dict[str, Any]]:
    with open(options_path, "r") as f:
        options_data = json.load(f)
    if isinstance(options_data, dict):
        options_list = flatten_hierarchical_options(options_data)
    elif isinstance(options_data, list):
        options_list = options_data
    else:
        raise ValueError(f"Invalid options type: {type(options_data)}")

    return [{"id": str(i), "description": desc} for i, desc in enumerate(options_list)]


def load_roles(
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


def load_utilities(utilities_path: str) -> Dict[str, float]:
    with open(utilities_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "utilities" in data:
        data = data["utilities"]

    if not isinstance(data, dict):
        raise ValueError(
            "utilities.json must be a dict mapping option_id -> utility (or a compute_utilities results JSON)."
        )

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


def parse_layers_spec(layers_spec: str, num_layers: int) -> List[int]:
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


def parse_rating(text: str) -> Optional[int]:
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


def models_yaml_path_for_experiment() -> str:
    """Path to utility_analysis/models.yaml (this file lives in …/linear_probes/lp/)."""
    here = os.path.dirname(os.path.abspath(__file__))
    utility_analysis = os.path.abspath(os.path.join(here, "..", "..", ".."))
    return os.path.join(utility_analysis, "models.yaml")
