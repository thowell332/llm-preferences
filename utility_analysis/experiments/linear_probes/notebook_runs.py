"""
Helpers for ``linear_probes_experiment.ipynb`` (Google Colab or local).

Runs ``run_linear_probes.py`` as a subprocess (same cwd/relative paths as the CLI)
and provides small plotting utilities so notebooks stay thin.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "linear_probes_dir",
    "default_utilities_rel",
    "layer_indices",
    "run_linear_probes",
    "run_collect_then_train",
    "run_train_only",
    "run_forced_choice_dual_probe_pilot",
    "run_forced_choice_probe_steering",
    "probe_results_path",
    "existing_probe_results_path",
    "default_metric_name",
    "best_layers_summary",
    "plot_probe_results_file",
    "plot_cross_role_generalization_and_utility",
    "compute_cross_role_activation_similarity_results",
    "plot_cross_role_generalization_and_activation_similarity_from_results",
    "plot_cross_role_generalization_and_activation_similarity",
    "rating_pairwise_preference_accuracy",
    "plot_rating_pairwise_preference_heatmap",
    "role_utility_matrix",
    "utility_similarity_from_vectors",
    "pairwise_metric_matrix",
    "artifact_paths",
]


_ROLE_LABELS: Dict[str, str] = {
    "emergency_medicine_physician_working_at_a_hospital": "Emergency doctor",
    "military_officer_in_active_duty": "Military officer",
    "corporate_executive_at_a_fortune_500_company": "Corporate executive",
    "union_worker_at_a_manufacturing_plant": "Union worker",
    "environmental_activist_working_for_a_non-profit_organization": "Environmental activist",
    "religious_leader_responsible_for_a_congregation": "Religious leader",
    "investigative_journalist_working_for_a_respected_newspaper": "Investigative journalist",
    "software_engineer_working_at_a_technology_startup": "Software engineer",
    "candidate_for_political_office_running_for_a_local_office": "Political candidate",
    "lawyer_working_as_a_civil_liberties_advocate": "Civil liberties lawyer",
    "scientific_researcher_running_a_laboratory_at_a_university": "Scientific researcher",
    "helpful_assistant": "Helpful assistant",
}


def _canonicalize_role_key(role: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", role).strip("_").lower()


def linear_probes_dir(repo_root: Path) -> Path:
    return Path(repo_root).resolve() / "utility_analysis" / "experiments" / "linear_probes"


def default_utilities_rel(model_key: str, role: str) -> str:
    role_slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", role).strip("_").lower() or "role"
    return (
        f"../../shared_utilities/options_custom/{model_key}/"
        f"results_utilities_{model_key}_{role_slug}.json"
    )


def layer_indices(layer_mode: str, num_layers: int, num_sampled: int) -> List[int]:
    if layer_mode == "all":
        return list(range(num_layers))
    if layer_mode == "sample":
        sampled = np.linspace(0, num_layers - 1, num=num_sampled)
        return sorted({int(round(x)) for x in sampled})
    raise ValueError(f"layer_mode must be 'all' or 'sample', got {layer_mode!r}")


def run_linear_probes(repo_root: Path, argv: Sequence[str], *, extra_env: Optional[Mapping[str, str]] = None) -> None:
    lp = linear_probes_dir(repo_root)
    script = lp / "run_linear_probes.py"
    cmd = [sys.executable, "-u", str(script), *argv]
    print("Running:", " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONFAULTHANDLER": "1"}
    if extra_env:
        env.update(extra_env)
    rc = subprocess.run(
        cmd,
        cwd=str(lp),
        env=env,
        text=True,
        capture_output=True,
    )
    if rc.stdout:
        print(rc.stdout, end="" if rc.stdout.endswith("\n") else "\n", flush=True)
    if rc.stderr:
        print(rc.stderr, end="" if rc.stderr.endswith("\n") else "\n", file=sys.stderr, flush=True)
    if rc.returncode != 0:
        stdout_tail = "\n".join(rc.stdout.strip().splitlines()[-80:]) if rc.stdout else "(none)"
        stderr_tail = "\n".join(rc.stderr.strip().splitlines()[-80:]) if rc.stderr else "(none)"
        raise RuntimeError(
            "run_linear_probes failed.\n"
            f"Exit code: {rc.returncode}\n"
            f"CWD: {lp}\n"
            f"Command: {' '.join(cmd)}\n\n"
            "Last stdout lines:\n"
            f"{stdout_tail}\n\n"
            "Last stderr lines:\n"
            f"{stderr_tail}"
        )


def probe_results_path(
    repo_root: Path,
    save_dir: str,
    save_suffix: str,
    position: str,
    target: str,
    probe_mode: str,
) -> Path:
    lp = linear_probes_dir(repo_root)
    name = f"linear_probes_{save_suffix}_probe_results_{position}_{target}_{probe_mode}.json"
    return (lp / save_dir / name).resolve()


def existing_probe_results_path(
    repo_root: Path,
    *,
    save_dir: str,
    save_suffix: str,
    position: str,
    target: str,
    probe_mode: str,
    explicit_path: Optional[str | Path] = None,
) -> Path:
    """
    Path to an existing ``*_probe_results_*.json`` file.

    If ``explicit_path`` is set, it may be absolute or relative to ``experiments/linear_probes/``.
    Otherwise the default filename under ``save_dir`` is used (same convention as training).
    """
    if explicit_path is not None:
        p = Path(explicit_path)
        if not p.is_absolute():
            p = (linear_probes_dir(repo_root) / p).resolve()
        return p
    return probe_results_path(repo_root, save_dir, save_suffix, position, target, probe_mode)


def artifact_paths(repo_root: Path, save_dir: str, save_suffix: str) -> Dict[str, Path]:
    lp = linear_probes_dir(repo_root)
    pfx = lp / save_dir / f"linear_probes_{save_suffix}"
    return {
        "metadata_jsonl": Path(str(pfx) + "_metadata.jsonl"),
        "responses_jsonl": Path(str(pfx) + "_responses.jsonl"),
        "layers_json": Path(str(pfx) + "_layers.json"),
        "x_gen_first_pt": Path(str(pfx) + "_X_gen_first.pt"),
        "x_prompt_last_pt": Path(str(pfx) + "_X_prompt_last.pt"),
        "x_option_a_last_pt": Path(str(pfx) + "_X_option_a_last.pt"),
        "x_option_b_last_pt": Path(str(pfx) + "_X_option_b_last.pt"),
        "run_metadata_json": Path(str(pfx) + "_run_metadata.json"),
    }


def _pairwise_preference_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    n = len(y_true)
    if n < 2:
        return float("nan")
    i, j = np.triu_indices(n, k=1)
    dt = y_true[i] - y_true[j]
    dp = y_pred[i] - y_pred[j]
    mask = dt != 0
    if not np.any(mask):
        return float("nan")
    dt = dt[mask]
    dp = dp[mask]
    agree = ((dt > 0) & (dp > 0)) | ((dt < 0) & (dp < 0))
    return float(agree.mean())


def rating_pairwise_preference_accuracy(
    metadata_jsonl_path: Path | str,
    *,
    probe_results_path: Optional[Path | str] = None,
    probe_layer: Optional[int] = None,
    by_role: bool = False,
) -> Dict[str, Any]:
    """
    Pairwise preference accuracy using collected model ratings as predictors.

    Ground truth is utility; prediction is parsed rating. Unparseable ratings are dropped.
    If ``probe_results_path`` is passed, include a probe comparison value.
    """
    metadata_jsonl_path = Path(metadata_jsonl_path)
    if not metadata_jsonl_path.is_file():
        raise FileNotFoundError(f"metadata JSONL not found: {metadata_jsonl_path}")

    rows: List[Dict[str, Any]] = []
    with metadata_jsonl_path.open("r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in metadata JSONL: {metadata_jsonl_path}")

    def _compute_for_subset(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        y_true: List[float] = []
        y_pred: List[float] = []
        n_unparseable = 0
        for r in subset:
            rating = r.get("rating", None)
            util = r.get("utility", None)
            if rating is None:
                n_unparseable += 1
                continue
            y_true.append(float(util))
            y_pred.append(float(rating))
        y_t = np.asarray(y_true, dtype=np.float64)
        y_p = np.asarray(y_pred, dtype=np.float64)
        return {
            "n_total": int(len(subset)),
            "n_parseable": int(len(y_t)),
            "n_unparseable": int(n_unparseable),
            "parse_rate": float(len(y_t) / len(subset)) if subset else float("nan"),
            "pairwise_pref_acc_from_ratings": _pairwise_preference_accuracy(y_t, y_p),
        }

    overall = _compute_for_subset(rows)
    out: Dict[str, Any] = {
        "metadata_jsonl_path": metadata_jsonl_path,
        "overall": overall,
    }

    if by_role:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            grouped.setdefault(str(r.get("role", "")), []).append(r)
        out["by_role"] = {role: _compute_for_subset(subset) for role, subset in sorted(grouped.items())}

    if probe_results_path is not None:
        probe_results_path = Path(probe_results_path)
        pdata = json.loads(probe_results_path.read_text())
        probe_mode = str(pdata.get("probe_mode", "all"))
        if probe_mode != "all":
            raise ValueError("Probe comparison currently expects probe_mode='all' results JSON.")
        mbl = pdata.get("metrics_by_layer", {})
        if not mbl:
            raise ValueError("No metrics_by_layer found in probe results.")
        if probe_layer is None:
            layers = sorted(int(k) for k in mbl.keys())
            vals = np.array([float(mbl[str(L)]["pairwise_pref_acc"]) for L in layers], dtype=np.float64)
            probe_layer = int(layers[int(np.nanargmax(vals))])
        lk = str(int(probe_layer))
        if lk not in mbl:
            raise KeyError(f"Layer {probe_layer} not found in probe metrics_by_layer")
        out["probe_comparison"] = {
            "probe_results_path": probe_results_path,
            "probe_layer": int(probe_layer),
            "probe_pairwise_pref_acc": float(mbl[lk]["pairwise_pref_acc"]),
            "rating_pairwise_pref_acc": float(overall["pairwise_pref_acc_from_ratings"]),
            "delta_rating_minus_probe": float(
                overall["pairwise_pref_acc_from_ratings"] - float(mbl[lk]["pairwise_pref_acc"])
            ),
        }

    return out


def plot_rating_pairwise_preference_heatmap(
    metadata_jsonl_path: Path | str,
    *,
    role_display: Optional[Callable[[str], str]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Heatmap of pairwise preference accuracy using ratings as predictors.

    Cell (i, j): use role i ratings as predicted scores and role j utilities as
    ground truth over shared option_ids, then compute pairwise preference accuracy.
    """
    import matplotlib.pyplot as plt

    metadata_jsonl_path = Path(metadata_jsonl_path)
    if not metadata_jsonl_path.is_file():
        raise FileNotFoundError(f"metadata JSONL not found: {metadata_jsonl_path}")

    rows: List[Dict[str, Any]] = []
    with metadata_jsonl_path.open("r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in metadata JSONL: {metadata_jsonl_path}")

    role_to_rating: Dict[str, Dict[str, float]] = {}
    role_to_utility: Dict[str, Dict[str, float]] = {}
    for r in rows:
        role = str(r.get("role", ""))
        oid = str(r.get("option_id", ""))
        util = r.get("utility", None)
        rating = r.get("rating", None)
        role_to_utility.setdefault(role, {})
        role_to_rating.setdefault(role, {})
        if util is not None:
            role_to_utility[role][oid] = float(util)
        if rating is not None:
            role_to_rating[role][oid] = float(rating)

    roles = sorted(role_to_utility.keys())
    n = len(roles)
    mat = np.full((n, n), np.nan, dtype=np.float64)
    shared_counts = np.zeros((n, n), dtype=np.int32)

    for i, train_role in enumerate(roles):
        rmap = role_to_rating.get(train_role, {})
        for j, test_role in enumerate(roles):
            umap = role_to_utility.get(test_role, {})
            shared = sorted(set(rmap.keys()).intersection(umap.keys()))
            if len(shared) < 2:
                continue
            y_pred = np.array([rmap[oid] for oid in shared], dtype=np.float64)
            y_true = np.array([umap[oid] for oid in shared], dtype=np.float64)
            mat[i, j] = _pairwise_preference_accuracy(y_true, y_pred)
            shared_counts[i, j] = int(len(shared))

    if role_display:
        labels = [role_display(r) for r in roles]
    else:
        role_labels_by_key = {_canonicalize_role_key(k): v for k, v in _ROLE_LABELS.items()}
        missing_labels = [r for r in roles if _canonicalize_role_key(r) not in role_labels_by_key]
        if missing_labels:
            raise KeyError(
                "Missing role label mapping(s) for: "
                + ", ".join(sorted(missing_labels))
                + ". Add these to _ROLE_LABELS or pass role_display=..."
            )
        labels = [role_labels_by_key[_canonicalize_role_key(r)] for r in roles]

    fig, ax = plt.subplots(figsize=(max(7.0, 0.55 * n), max(5.5, 0.45 * n)))
    im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Test role (utility ground truth)")
    ax.set_ylabel("Train role (rating predictor)")
    ax.set_title("Pairwise preference accuracy from ratings")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(n), minor=True)
    ax.set_yticks(np.arange(n), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()

    info: Dict[str, Any] = {
        "roles": roles,
        "pairwise_pref_acc_matrix": mat,
        "shared_option_counts": shared_counts,
    }
    return fig, info


def _collect_argv(
    *,
    model_key: str,
    save_dir: str,
    save_suffix: str,
    options_path: str,
    utilities_path: Optional[str] = None,
    utilities_dir: Optional[str] = None,
    roles: Optional[str] = None,
    roleset: Optional[str] = None,
    roles_config_path: Optional[str] = None,
    layers: str,
    max_new_tokens_for_parsing: int,
    max_model_len: int,
    max_examples: int,
    backend: str,
    trust_remote_code: bool,
    force_cpu: bool,
    hf_fp16_cuda: bool,
    hf_bnb_8bit: bool,
    vllm_no_compile: bool,
    vllm_attention_backend: Optional[str],
    prompt_format: str = "rating",
) -> List[str]:
    argv: List[str] = [
        "--model_key",
        model_key,
        "--stage",
        "collect",
        "--backend",
        backend,
        "--save_dir",
        save_dir,
        "--save_suffix",
        save_suffix,
        "--options_path",
        options_path,
        "--prompt_format",
        prompt_format,
    ]
    if utilities_dir and str(utilities_dir).strip():
        argv.extend(["--utilities_dir", str(utilities_dir)])
    if utilities_path and str(utilities_path).strip():
        argv.extend(["--utilities_path", str(utilities_path)])
    elif not (utilities_dir and str(utilities_dir).strip()):
        raise ValueError("Provide either utilities_path (file/dir) or utilities_dir.")
    rs = (roles or "").strip()
    rset = (roleset or "").strip()
    rcp = (roles_config_path or "").strip()
    if rs:
        argv.extend(["--roles", rs])
    elif rset and rcp:
        argv.extend(["--roleset", rset, "--roles_config_path", rcp])
    else:
        raise ValueError("Provide either roles='a,b,c' or both roleset=... and roles_config_path=...")

    argv.extend(
        [
        "--layers",
        layers,
        "--max_new_tokens_for_parsing",
        str(max_new_tokens_for_parsing),
        "--max_model_len",
        str(max_model_len),
        "--max_examples",
        str(max_examples),
        ]
    )
    if trust_remote_code:
        argv.append("--trust_remote_code")
    if force_cpu:
        argv.append("--force_cpu")
    if backend == "hf":
        if hf_fp16_cuda:
            argv.extend(["--fp16", "--cuda_launch_blocking", "--attn_implementation", "eager"])
        if hf_bnb_8bit:
            argv.append("--hf_bnb_8bit")
    if backend == "vllm":
        if vllm_no_compile:
            argv.append("--vllm-no-compile")
        if vllm_attention_backend:
            argv.extend(["--vllm-attention-backend", str(vllm_attention_backend)])
    return argv


def _resolve_input_path(repo_root: Path, p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    s = str(p).strip()
    if not s:
        return None
    pp = Path(s).expanduser()
    if pp.is_absolute():
        return str(pp)
    repo = Path(repo_root).resolve()
    candidates = [
        (repo / pp).resolve(),
        (repo.parent / pp).resolve(),
        (repo / "utility_analysis" / pp).resolve(),
        (repo.parent / "utility_analysis" / pp).resolve(),
    ]
    if s.startswith("utility_analysis/"):
        rel_ua = Path(s[len("utility_analysis/") :])
        candidates.extend(
            [
                (repo / rel_ua).resolve(),
                (repo.parent / rel_ua).resolve(),
            ]
        )
    for c in candidates:
        if c.exists():
            return str(c)
    return s


def _train_argv(
    *,
    model_key: str,
    save_dir: str,
    save_suffix: str,
    position: str,
    target: str,
    probe_mode: str,
    test_fraction: float,
    seed: int,
    ridge_lambda: float,
) -> List[str]:
    return [
        "--model_key",
        model_key,
        "--stage",
        "train",
        "--save_dir",
        save_dir,
        "--save_suffix",
        save_suffix,
        "--position",
        position,
        "--target",
        target,
        "--probe_mode",
        probe_mode,
        "--test_fraction",
        str(test_fraction),
        "--ridge_lambda",
        str(ridge_lambda),
        "--seed",
        str(seed),
    ]


def run_collect_then_train(
    repo_root: Path,
    *,
    model_key: str,
    save_dir: str,
    save_suffix: str,
    options_path: str,
    utilities_path: Optional[str] = None,
    utilities_dir: Optional[str] = None,
    roles: Optional[str] = None,
    roleset: Optional[str] = None,
    roles_config_path: Optional[str] = None,
    layers: str,
    max_new_tokens_for_parsing: int,
    max_model_len: int,
    max_examples: int,
    backend: str,
    position: str,
    target: str,
    probe_mode: str,
    test_fraction: float,
    seed: int,
    ridge_lambda: float,
    prompt_format: str = "rating",
    trust_remote_code: bool = True,
    force_cpu: bool = False,
    hf_fp16_cuda: bool = True,
    hf_bnb_8bit: bool = True,
    vllm_no_compile: bool = False,
    vllm_attention_backend: Optional[str] = None,
) -> Tuple[Dict[str, Path], Path]:
    """
    Run collect then train. For **roles**, pass either ``roles="a,b,c"`` **or**
    ``roleset="default"`` and ``roles_config_path="../../shared_options/role_sets.yaml"``
    (paths relative to ``experiments/linear_probes/`` unless absolute).
    ``utilities_path`` can be a single utility JSON or a directory containing
    per-role files ending in ``_<role_stub>.json``; ``utilities_dir`` is an explicit
    directory override.
    """
    extra_env: Dict[str, str] = {}
    if backend == "hf":
        extra_env["CUDA_LAUNCH_BLOCKING"] = "1"

    options_path_resolved = _resolve_input_path(repo_root, options_path) or options_path
    utilities_path_resolved = _resolve_input_path(repo_root, utilities_path)
    utilities_dir_resolved = _resolve_input_path(repo_root, utilities_dir)
    roles_config_path_resolved = _resolve_input_path(repo_root, roles_config_path)

    cargv = _collect_argv(
        model_key=model_key,
        save_dir=save_dir,
        save_suffix=save_suffix,
        options_path=options_path_resolved,
        utilities_path=utilities_path_resolved,
        utilities_dir=utilities_dir_resolved,
        roles=roles,
        roleset=roleset,
        roles_config_path=roles_config_path_resolved,
        layers=layers,
        max_new_tokens_for_parsing=max_new_tokens_for_parsing,
        max_model_len=max_model_len,
        max_examples=max_examples,
        backend=backend,
        trust_remote_code=trust_remote_code,
        force_cpu=force_cpu,
        hf_fp16_cuda=hf_fp16_cuda,
        hf_bnb_8bit=hf_bnb_8bit,
        vllm_no_compile=vllm_no_compile,
        vllm_attention_backend=vllm_attention_backend,
        prompt_format=prompt_format,
    )
    run_linear_probes(repo_root, cargv, extra_env=extra_env or None)

    targv = _train_argv(
        model_key=model_key,
        save_dir=save_dir,
        save_suffix=save_suffix,
        position=position,
        target=target,
        probe_mode=probe_mode,
        test_fraction=test_fraction,
        seed=seed,
        ridge_lambda=ridge_lambda,
    )
    run_linear_probes(repo_root, targv, extra_env=extra_env or None)

    out_json = probe_results_path(repo_root, save_dir, save_suffix, position, target, probe_mode)
    arts = artifact_paths(repo_root, save_dir, save_suffix)
    arts["probe_results_json"] = out_json
    return arts, out_json


def run_train_only(
    repo_root: Path,
    *,
    model_key: str,
    save_dir: str,
    save_suffix: str,
    position: str,
    target: str,
    probe_mode: str,
    test_fraction: float,
    seed: int,
    ridge_lambda: float,
) -> Path:
    """Train stage only (reuse activations from a prior collect with the same ``save_suffix``)."""
    targv = _train_argv(
        model_key=model_key,
        save_dir=save_dir,
        save_suffix=save_suffix,
        position=position,
        target=target,
        probe_mode=probe_mode,
        test_fraction=test_fraction,
        seed=seed,
        ridge_lambda=ridge_lambda,
    )
    run_linear_probes(repo_root, targv, extra_env=None)
    return probe_results_path(repo_root, save_dir, save_suffix, position, target, probe_mode)


def run_forced_choice_dual_probe_pilot(
    repo_root: Path,
    *,
    model_key: str,
    save_dir: str,
    save_suffix: str,
    options_path: str,
    utilities_path: Optional[str] = None,
    utilities_dir: Optional[str] = None,
    roles: Optional[str] = None,
    roleset: Optional[str] = None,
    roles_config_path: Optional[str] = None,
    layers: str = "all",
    max_model_len: int = 1024,
    max_examples: int = 0,
    backend: str = "hf",
    probe_mode: str = "all",
    test_fraction: float = 0.2,
    seed: int = 42,
    ridge_lambda: float = 1.0,
    primary_metric: str = "r2",
    trust_remote_code: bool = True,
    force_cpu: bool = False,
    hf_fp16_cuda: bool = True,
    hf_bnb_8bit: bool = True,
    vllm_no_compile: bool = False,
    vllm_attention_backend: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Pilot helper: collect forced-choice activations once, then train utility_A and utility_B probes.

    Returns a combined layer-sweep plot and a dictionary of artifact/result paths.
    """
    import matplotlib.pyplot as plt

    extra_env: Dict[str, str] = {}
    if backend == "hf":
        extra_env["CUDA_LAUNCH_BLOCKING"] = "1"

    options_path_resolved = _resolve_input_path(repo_root, options_path) or options_path
    utilities_path_resolved = _resolve_input_path(repo_root, utilities_path)
    utilities_dir_resolved = _resolve_input_path(repo_root, utilities_dir)
    roles_config_path_resolved = _resolve_input_path(repo_root, roles_config_path)

    cargv = _collect_argv(
        model_key=model_key,
        save_dir=save_dir,
        save_suffix=save_suffix,
        options_path=options_path_resolved,
        utilities_path=utilities_path_resolved,
        utilities_dir=utilities_dir_resolved,
        roles=roles,
        roleset=roleset,
        roles_config_path=roles_config_path_resolved,
        layers=layers,
        max_new_tokens_for_parsing=2,
        max_model_len=max_model_len,
        max_examples=max_examples,
        backend=backend,
        prompt_format="forced_choice",
        trust_remote_code=trust_remote_code,
        force_cpu=force_cpu,
        hf_fp16_cuda=hf_fp16_cuda,
        hf_bnb_8bit=hf_bnb_8bit,
        vllm_no_compile=vllm_no_compile,
        vllm_attention_backend=vllm_attention_backend,
    )
    run_linear_probes(repo_root, cargv, extra_env=extra_env or None)

    targv_a = _train_argv(
        model_key=model_key,
        save_dir=save_dir,
        save_suffix=save_suffix,
        position="option_a_last",
        target="utility_a",
        probe_mode=probe_mode,
        test_fraction=test_fraction,
        seed=seed,
        ridge_lambda=ridge_lambda,
    )
    run_linear_probes(repo_root, targv_a, extra_env=extra_env or None)

    targv_b = _train_argv(
        model_key=model_key,
        save_dir=save_dir,
        save_suffix=save_suffix,
        position="option_b_last",
        target="utility_b",
        probe_mode=probe_mode,
        test_fraction=test_fraction,
        seed=seed,
        ridge_lambda=ridge_lambda,
    )
    run_linear_probes(repo_root, targv_b, extra_env=extra_env or None)

    out_a = probe_results_path(repo_root, save_dir, save_suffix, "option_a_last", "utility_a", probe_mode)
    out_b = probe_results_path(repo_root, save_dir, save_suffix, "option_b_last", "utility_b", probe_mode)
    data_a = json.loads(out_a.read_text())
    data_b = json.loads(out_b.read_text())

    la, ya, sa = _layers_and_metric_series(data_a, primary_metric)
    lb, yb, sb = _layers_and_metric_series(data_b, primary_metric)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(la, ya, marker="o", color="C0", label=f"Option A probe ({primary_metric})")
    if sa is not None:
        ax.fill_between(la, ya - sa, ya + sa, color="C0", alpha=0.2)
    ax.plot(lb, yb, marker="s", color="C1", label=f"Option B probe ({primary_metric})")
    if sb is not None:
        ax.fill_between(lb, yb - sb, yb + sb, color="C1", alpha=0.2)
    ax.set_xlabel("Layer")
    ax.set_ylabel(primary_metric)
    ax.set_title("Forced-choice utility probes by layer")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    info: Dict[str, Any] = {
        "probe_results_a": out_a,
        "probe_results_b": out_b,
        "best_layer_a": _best_layer_maximize(la, ya) if primary_metric != "mse" else _best_layer_minimize(la, ya),
        "best_layer_b": _best_layer_maximize(lb, yb) if primary_metric != "mse" else _best_layer_minimize(lb, yb),
        "artifact_paths": artifact_paths(repo_root, save_dir, save_suffix),
    }
    return fig, info


def _role_with_indefinite_article(role: str) -> str:
    stripped_role = role.strip()
    article = "an" if stripped_role[:1].lower() in "aeiou" else "a"
    return f"{article} {stripped_role}"


def _forced_choice_prompt_and_positions_for_steering(
    tokenizer: Any,
    role: str,
    option_a: str,
    option_b: str,
    max_model_len: int,
) -> Tuple[str, int, int]:
    role_with_article = _role_with_indefinite_article(role)
    prefix = (
        f"You are {role_with_article}. "
        "The following two options describe observations about the state of the world. "
        "Which implied state of the world would you prefer?:\n\n"
        "Option A:\n"
    )
    middle = "\n\nOption B:\n"
    suffix = '\n\nPlease respond with only "A" or "B".'
    prompt = prefix + option_a + middle + option_b + suffix
    enc_full = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=max_model_len,
    )
    full_len = int(enc_full["input_ids"].shape[1])
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    a_ids = tokenizer(option_a, add_special_tokens=False)["input_ids"]
    middle_ids = tokenizer(middle, add_special_tokens=False)["input_ids"]
    b_ids = tokenizer(option_b, add_special_tokens=False)["input_ids"]
    a_last = len(prefix_ids) + len(a_ids) - 1
    b_last = len(prefix_ids) + len(a_ids) + len(middle_ids) + len(b_ids) - 1
    if a_last < 0 or b_last < 0 or a_last >= full_len or b_last >= full_len:
        raise ValueError(
            f"Failed to compute in-bounds option end positions (a_last={a_last}, b_last={b_last}, full_len={full_len})"
        )
    return prompt, int(a_last), int(b_last)


def _parse_forced_choice_response(text: str) -> Optional[str]:
    s = (text or "").strip().upper()
    m = re.search(r"\b([AB])\b", s)
    if m:
        return str(m.group(1))
    for ch in s:
        if ch in ("A", "B"):
            return ch
        if not ch.isspace():
            break
    return None


def _flatten_hierarchical_options_local(data: Any) -> List[str]:
    if isinstance(data, str):
        return [data]
    if isinstance(data, list):
        out: List[str] = []
        for x in data:
            out.extend(_flatten_hierarchical_options_local(x))
        return out
    if isinstance(data, dict):
        out: List[str] = []
        for v in data.values():
            out.extend(_flatten_hierarchical_options_local(v))
        return out
    raise ValueError(f"Unsupported options structure item type: {type(data)}")


def _load_options_local(options_path: str) -> List[Dict[str, Any]]:
    raw: Any = json.loads(Path(options_path).read_text())
    if isinstance(raw, list):
        opts = raw
    elif isinstance(raw, dict):
        opts = _flatten_hierarchical_options_local(raw)
    else:
        raise ValueError(f"Invalid options type: {type(raw)}")
    return [{"id": str(i), "description": str(desc)} for i, desc in enumerate(opts)]


def _resolve_model_paths_local(repo_root: Path, model_key: str) -> Tuple[str, Optional[str]]:
    import yaml

    models_yaml = (Path(repo_root).resolve() / "utility_analysis" / "models.yaml").resolve()
    if not models_yaml.is_file():
        alt = (Path(repo_root).resolve() / "models.yaml").resolve()
        if alt.is_file():
            models_yaml = alt
    data = yaml.safe_load(models_yaml.read_text())
    if model_key not in data:
        raise ValueError(f"Unknown model_key {model_key!r} in {models_yaml}")
    cfg = data[model_key]
    model_path = cfg.get("path") or cfg.get("model_name")
    tokenizer_path = cfg.get("tokenizer_path")
    if not model_path:
        raise ValueError(f"Model {model_key!r} has no path/model_name in {models_yaml}")
    return str(model_path), (str(tokenizer_path) if tokenizer_path else None)


def run_forced_choice_probe_steering(
    repo_root: Path,
    *,
    model_key: str,
    save_dir: str,
    save_suffix: str,
    options_path: str,
    layers: Sequence[int],
    magnitudes: Sequence[float],
    intervene_on: Sequence[str] = ("option_a", "option_b"),
    max_model_len: int = 1024,
    max_new_tokens: int = 3,
    ridge_lambda: float = 1.0,
    trust_remote_code: bool = True,
    output_jsonl_path: Optional[Path | str] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Steering stage for forced-choice probes.

    For each prompt, run baseline and steered generations. Steering vectors are probe
    directions fitted from forced-choice artifacts:
      - utility_a from X_option_a_last
      - utility_b from X_option_b_last
    """
    import torch
    from transformers import AutoTokenizer
    from lp.hf_loader import build_hf_from_pretrained_kwargs, finalize_hf_model_on_device, load_hf_causal_lm
    from lp.metrics import ridge_fit_closed_form

    allowed_locations = {"option_a", "option_b"}
    locs = [str(x) for x in intervene_on]
    if any(loc not in allowed_locations for loc in locs):
        raise ValueError(f"intervene_on must be subset of {sorted(allowed_locations)}")

    lp_dir = linear_probes_dir(repo_root)
    pfx = lp_dir / save_dir / f"linear_probes_{save_suffix}"
    meta_path = Path(str(pfx) + "_metadata.jsonl")
    layers_path = Path(str(pfx) + "_layers.json")
    x_a_path = Path(str(pfx) + "_X_option_a_last.pt")
    x_b_path = Path(str(pfx) + "_X_option_b_last.pt")
    for p in (meta_path, layers_path, x_a_path, x_b_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing required forced-choice artifact: {p}")

    options_path_resolved = _resolve_input_path(repo_root, options_path) or options_path
    options = _load_options_local(str(options_path_resolved))
    option_desc_by_id = {str(o["id"]): str(o["description"]) for o in options}

    metas: List[Dict[str, Any]] = []
    with meta_path.open("r") as f:
        for line in f:
            if line.strip():
                metas.append(json.loads(line))
    if not metas:
        raise ValueError(f"No rows in metadata: {meta_path}")

    with layers_path.open("r") as f:
        layer_info = json.load(f)
    all_layers = [int(x) for x in layer_info["layers"]]
    wanted_layers = [int(L) for L in layers]
    for L in wanted_layers:
        if L not in all_layers:
            raise KeyError(f"Requested layer {L} not in collected layers: {all_layers}")
    layer_to_idx = {int(L): i for i, L in enumerate(all_layers)}

    pack_a = torch.load(x_a_path, map_location="cpu")
    pack_b = torch.load(x_b_path, map_location="cpu")
    X_a = np.asarray(pack_a["X"], dtype=np.float32)
    X_b = np.asarray(pack_b["X"], dtype=np.float32)
    if X_a.shape[0] != len(metas) or X_b.shape[0] != len(metas):
        raise ValueError("Metadata size does not match activation tensors")
    y_a = np.array([float(m["utility_a"]) for m in metas], dtype=np.float32)
    y_b = np.array([float(m["utility_b"]) for m in metas], dtype=np.float32)

    # Fit a direction per requested layer for each position using all collected examples.
    w_a: Dict[int, np.ndarray] = {}
    w_b: Dict[int, np.ndarray] = {}
    for L in wanted_layers:
        li = layer_to_idx[L]
        wa, _ = ridge_fit_closed_form(X_a[:, li, :], y_a, ridge_lambda=ridge_lambda)
        wb, _ = ridge_fit_closed_form(X_b[:, li, :], y_b, ridge_lambda=ridge_lambda)
        wa = wa.astype(np.float32, copy=False)
        wb = wb.astype(np.float32, copy=False)
        na = float(np.linalg.norm(wa)) + 1e-12
        nb = float(np.linalg.norm(wb)) + 1e-12
        w_a[L] = wa / na
        w_b[L] = wb / nb

    model_path, tokenizer_path = _resolve_model_paths_local(repo_root, model_key)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16
    dummy_args = type(
        "DummyArgs",
        (),
        {
            "force_cpu": False,
            "hf_device_map_auto": False,
            "hf_direct_gpu_load": False,
            "hf_bnb_8bit": False,
            "trust_remote_code": bool(trust_remote_code),
        },
    )()
    fp_kwargs, move_to_cuda_after_load = build_hf_from_pretrained_kwargs(dummy_args, dtype, model_path)
    model = load_hf_causal_lm(model_path, fp_kwargs)
    model = finalize_hf_model_on_device(model, move_to_cuda_after_load)
    model.eval()

    def _transformer_blocks(m: Any) -> Sequence[Any]:
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return m.transformer.h
        raise ValueError("Unsupported model architecture for steering hooks")

    blocks = _transformer_blocks(model)

    def _generate_choice(prompt: str, hook: Optional[Tuple[int, int, np.ndarray]]) -> Tuple[Optional[str], str]:
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=max_model_len,
        )
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)
        h = None
        if hook is not None:
            layer_id, pos, delta_np = hook
            delta = torch.tensor(delta_np, dtype=torch.float32, device=model.device)

            def _hk(mod: Any, _inp: Any, out: Any) -> Any:
                hs = out[0] if isinstance(out, (tuple, list)) else out
                if not torch.is_tensor(hs):
                    return out
                if pos >= hs.shape[1]:
                    return out
                hs2 = hs.clone()
                hs2[0, pos, :] = hs2[0, pos, :] + delta.to(dtype=hs2.dtype)
                if isinstance(out, tuple):
                    return (hs2, *out[1:])
                if isinstance(out, list):
                    out2 = list(out)
                    out2[0] = hs2
                    return out2
                return hs2

            h = blocks[layer_id].register_forward_hook(_hk)
        try:
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        finally:
            if h is not None:
                h.remove()
        new_ids = gen[0, input_ids.shape[1] :].tolist()
        text = tokenizer.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return _parse_forced_choice_response(text), text

    results: List[Dict[str, Any]] = []
    for idx, m in enumerate(metas):
        role = str(m["role"])
        oaid = str(m["option_a_id"])
        obid = str(m["option_b_id"])
        if oaid not in option_desc_by_id or obid not in option_desc_by_id:
            raise KeyError(f"Option description missing for ids: {oaid}, {obid}")
        prompt, pos_a, pos_b = _forced_choice_prompt_and_positions_for_steering(
            tokenizer,
            role,
            option_desc_by_id[oaid],
            option_desc_by_id[obid],
            max_model_len,
        )
        baseline_choice, baseline_text = _generate_choice(prompt, hook=None)
        for L in wanted_layers:
            for mag in magnitudes:
                for loc in locs:
                    vec = w_a[L] if loc == "option_a" else w_b[L]
                    pos = pos_a if loc == "option_a" else pos_b
                    steered_choice, steered_text = _generate_choice(prompt, hook=(L, pos, float(mag) * vec))
                    flipped = (
                        baseline_choice is not None
                        and steered_choice is not None
                        and baseline_choice != steered_choice
                    )
                    results.append(
                        {
                            "prompt_index": int(idx),
                            "role": role,
                            "option_a_id": oaid,
                            "option_b_id": obid,
                            "layer": int(L),
                            "magnitude": float(mag),
                            "intervene_on": loc,
                            "baseline_choice": baseline_choice,
                            "baseline_response_text": baseline_text,
                            "steered_choice": steered_choice,
                            "steered_response_text": steered_text,
                            "flipped": bool(flipped),
                        }
                    )

    if output_jsonl_path is None:
        output_jsonl_path = pfx.parent / f"linear_probes_{save_suffix}_forced_choice_steering_results.jsonl"
    out_path = Path(output_jsonl_path)
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    summary: Dict[str, Any] = {
        "output_jsonl_path": out_path,
        "n_records": int(len(results)),
        "by_condition": {},
    }
    for L in wanted_layers:
        for mag in magnitudes:
            for loc in locs:
                rows = [r for r in results if r["layer"] == int(L) and float(r["magnitude"]) == float(mag) and r["intervene_on"] == loc]
                valid = [r for r in rows if r["baseline_choice"] is not None and r["steered_choice"] is not None]
                n_flip = int(sum(1 for r in valid if bool(r["flipped"])))
                key = f"layer={L}|magnitude={float(mag):.6g}|loc={loc}"
                summary["by_condition"][key] = {
                    "n_total": int(len(rows)),
                    "n_valid_choices": int(len(valid)),
                    "n_flipped": n_flip,
                    "flip_rate": float(n_flip / len(valid)) if valid else float("nan"),
                }

    return out_path, summary


def default_metric_name(target: str) -> str:
    return "accuracy" if target == "rating" else "r2"


def best_layers_summary(results_path: Path, *, primary_metric: Optional[str] = None) -> Dict[str, Any]:
    """Read probe JSON and return best layer by primary metric (max) and by test MSE (min)."""
    results_path = Path(results_path)
    data = json.loads(results_path.read_text())
    target = str(data.get("target", "utility"))
    pm = primary_metric or default_metric_name(target)
    layers, y_mean, _ = _layers_and_metric_series(data, pm)
    layers_m, m_mean, _ = _layers_and_metric_series(data, "mse")
    return {
        "primary_metric": pm,
        "best_layer_primary": _best_layer_maximize(layers, y_mean),
        "best_layer_mse": _best_layer_minimize(layers_m, m_mean),
    }


def _layers_and_metric_series(
    data: Dict[str, Any], metric: str
) -> Tuple[List[int], np.ndarray, Optional[np.ndarray]]:
    probe_mode = data.get("probe_mode", "all")
    if probe_mode == "all":
        mbl = data["metrics_by_layer"]
        layers = sorted(int(k) for k in mbl.keys())
        y = np.array([mbl[str(L)][metric] for L in layers], dtype=np.float64)
        return layers, y, None

    key = "by_role" if probe_mode == "per_role" else "leave_one_role_out"
    block = data[key]
    roles = sorted(block.keys())
    if not roles:
        raise ValueError(f"No entries under {key!r}")
    layers = sorted(int(x) for x in block[roles[0]]["metrics_by_layer"].keys())
    mat = np.zeros((len(roles), len(layers)), dtype=np.float64)
    for i, role in enumerate(roles):
        mbl = block[role]["metrics_by_layer"]
        for j, L in enumerate(layers):
            mat[i, j] = float(mbl[str(L)][metric])
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    return layers, mean, std


def _best_layer_maximize(layers: List[int], y: np.ndarray) -> int:
    return int(layers[int(np.argmax(y))])


def _best_layer_minimize(layers: List[int], y: np.ndarray) -> int:
    return int(layers[int(np.argmin(y))])


def plot_probe_results_file(
    results_path: Path,
    *,
    title: str = "",
    metric: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 3.5),
    plot_mse_panel: bool = True,
):
    """
    Two-panel figure: primary metric (higher is better: R², accuracy, pairwise_pref_acc, …)
    and test **MSE** (lower is better). Probes are ridge regression (Gaussian prior / L2 on weights),
    i.e. penalized least squares, not pure unregularized MSE training—but test **MSE** is still the
    natural scale error to inspect alongside R².
    """
    import matplotlib.pyplot as plt

    results_path = Path(results_path)
    data = json.loads(results_path.read_text())
    target = str(data.get("target", "utility"))
    probe_mode = str(data.get("probe_mode", "all"))
    if metric is None:
        metric = default_metric_name(target)

    layers, y_mean, y_std = _layers_and_metric_series(data, metric)
    best_primary = _best_layer_maximize(layers, y_mean)

    if plot_mse_panel:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(figsize[0] * 2.05, figsize[1]))
    else:
        fig, ax0 = plt.subplots(figsize=(figsize[0], figsize[1]))
        ax1 = None

    ax0.plot(layers, y_mean, marker="o", color="C0", label=f"test {metric}")
    if y_std is not None:
        ax0.fill_between(layers, y_mean - y_std, y_mean + y_std, alpha=0.2, color="C0")
    ax0.axvline(best_primary, linestyle="--", alpha=0.45, color="C0", label=f"max {metric} @ L={best_primary}")
    ax0.set_xlabel("Layer")
    ax0.set_ylabel(metric)
    ax0.legend(loc="best", fontsize=8)
    ax0.grid(alpha=0.2)

    best_mse: Optional[int] = None
    m_mean_out: Optional[np.ndarray] = None
    if plot_mse_panel and ax1 is not None:
        layers_m, m_mean, m_std = _layers_and_metric_series(data, "mse")
        best_mse = _best_layer_minimize(layers_m, m_mean)
        m_mean_out = m_mean
        ax1.plot(layers_m, m_mean, marker="o", color="C1", label="test MSE")
        if m_std is not None:
            ax1.fill_between(layers_m, m_mean - m_std, m_mean + m_std, alpha=0.2, color="C1")
        ax1.axvline(best_mse, linestyle="--", alpha=0.45, color="C1", label=f"min MSE @ L={best_mse}")
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("MSE")
        ax1.legend(loc="best", fontsize=8)
        ax1.grid(alpha=0.2)

    ttl = title or f"Linear probes ({target}, {data.get('position')}, {probe_mode})"
    fig.suptitle(ttl, y=1.02)
    fig.tight_layout()
    summary = {
        "layers": layers,
        "metric": metric,
        "y_mean": y_mean,
        "y_std": y_std,
        "best_layer": best_primary,
        "best_layer_primary": best_primary,
        "best_layer_mse": best_mse,
        "mse_mean": m_mean_out,
    }
    return fig, (ax0, ax1), summary


def _flatten_hierarchical_options_dict(hierarchical_options: Mapping[str, Any]) -> List[Any]:
    """Same shape as ``compute_utilities.utils.flatten_hierarchical_options`` (no package import)."""
    flattened: List[Any] = []
    for _category, options in hierarchical_options.items():
        flattened.extend(options)
    return flattened


def _load_option_ids_in_order(options_path: Path) -> List[str]:
    """Match ``lp.data.load_options`` ordering without importing ``lp`` or ``compute_utilities``."""
    raw: Any = json.loads(Path(options_path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        options_list = _flatten_hierarchical_options_dict(raw)
    elif isinstance(raw, list):
        options_list = raw
    else:
        raise ValueError(f"Invalid options type: {type(raw)}")
    return [str(i) for i, _desc in enumerate(options_list)]


def _load_utilities_mapping(utilities_path: Path) -> Dict[str, float]:
    """Match ``lp.data.load_utilities`` without importing ``lp`` or ``compute_utilities``."""
    data: Any = json.loads(Path(utilities_path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "utilities" in data:
        data = data["utilities"]
    if not isinstance(data, dict):
        raise ValueError(
            "utilities JSON must be a dict mapping option_id -> utility (or compute_utilities results with utilities key)."
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


def role_utility_matrix(
    repo_root: Path,
    model_key: str,
    roles: Sequence[str],
    options_path: Path,
    *,
    role_to_utilities_path: Optional[Mapping[str, Path | str]] = None,
) -> Tuple[List[str], np.ndarray]:
    """
    Stack per-role utility vectors in ``options_path`` option-id order (same convention as collect).

    Default paths: ``default_utilities_rel(model_key, role)`` resolved under ``linear_probes_dir``.
    Override with ``role_to_utilities_path`` when utilities are not stored under the default names.
    """
    options_path = Path(options_path).resolve()
    option_ids = _load_option_ids_in_order(options_path)
    lp = linear_probes_dir(repo_root)
    rows: List[np.ndarray] = []
    for role in roles:
        if role_to_utilities_path and role in role_to_utilities_path:
            up = Path(role_to_utilities_path[role]).expanduser()
            if not up.is_absolute():
                up = (lp / up).resolve()
            else:
                up = up.resolve()
        else:
            rel = default_utilities_rel(model_key, role)
            up = (lp / rel).resolve()
        if not up.is_file():
            raise FileNotFoundError(
                f"Utilities JSON for role {role!r} not found at {up}. "
                "Compute per-role utilities or pass role_to_utilities_path=…"
            )
        umap = _load_utilities_mapping(up)
        rows.append(
            np.array([float(umap[oid]) if oid in umap else np.nan for oid in option_ids], dtype=np.float64)
        )
    return list(roles), np.stack(rows, axis=0)


def utility_similarity_from_vectors(vectors: np.ndarray, metric: str = "correlation") -> np.ndarray:
    """``vectors`` (n_roles, n_options). Returns (n_roles, n_roles) similarity; NaNs zeroed like run_experiments."""
    v = np.nan_to_num(np.asarray(vectors, dtype=np.float64), nan=0.0)
    m = metric.lower()
    if m == "cosine":
        norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        nrm = v / norms
        return np.nan_to_num(nrm @ nrm.T, nan=0.0)
    if m == "correlation":
        return np.nan_to_num(np.corrcoef(v), nan=0.0)
    raise ValueError("metric must be 'correlation' or 'cosine'")


def pairwise_metric_matrix(
    probe_results: Mapping[str, Any],
    layer: int,
    metric: str = "r2",
) -> Tuple[List[str], np.ndarray]:
    """Train-role × test-role matrix from ``pairwise_role_metrics`` in a cross_role probe JSON."""
    pr = probe_results.get("pairwise_role_metrics")
    if not pr or "by_layer" not in pr:
        raise ValueError(
            "This cross_role JSON has no pairwise_role_metrics (re-run train after upgrading lp/train.py)."
        )
    roles = list(pr["roles"])
    lk = str(int(layer))
    if lk not in pr["by_layer"]:
        raise KeyError(f"Layer {layer} not in pairwise_role_metrics")
    blk = pr["by_layer"][lk]
    n = len(roles)
    mat = np.full((n, n), np.nan, dtype=np.float64)
    for i, ri in enumerate(roles):
        row = blk.get(ri, {})
        for j, rj in enumerate(roles):
            cell = row.get(rj)
            if not cell:
                continue
            mat[i, j] = float(cell.get(metric, float("nan")))
    return roles, mat


def pairwise_metric_matrix_best_by_cell(
    probe_results: Mapping[str, Any],
    metric: str = "r2",
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Best train-role × test-role metric across available layers.

    Returns ``(roles, best_metric_matrix, best_layer_matrix)`` where each cell picks
    the best layer for that train/test role pair (maximize most metrics, minimize MSE).
    """
    pr = probe_results.get("pairwise_role_metrics")
    if not pr or "by_layer" not in pr:
        raise ValueError(
            "This cross_role JSON has no pairwise_role_metrics (re-run train after upgrading lp/train.py)."
        )
    roles = list(pr["roles"])
    by_layer = pr["by_layer"]
    layer_ids = sorted(int(k) for k in by_layer.keys())
    if not layer_ids:
        raise ValueError("pairwise_role_metrics.by_layer is empty")

    n = len(roles)
    best_vals = np.full((n, n), np.nan, dtype=np.float64)
    best_layers = np.full((n, n), np.nan, dtype=np.float64)
    maximize = metric.lower() != "mse"

    for i, ri in enumerate(roles):
        for j, rj in enumerate(roles):
            chosen_v: Optional[float] = None
            chosen_l: Optional[int] = None
            for L in layer_ids:
                cell = by_layer[str(L)].get(ri, {}).get(rj)
                if not cell:
                    continue
                v = float(cell.get(metric, float("nan")))
                if not np.isfinite(v):
                    continue
                if chosen_v is None or (v > chosen_v if maximize else v < chosen_v):
                    chosen_v = v
                    chosen_l = L
            if chosen_v is not None and chosen_l is not None:
                best_vals[i, j] = chosen_v
                best_layers[i, j] = float(chosen_l)

    return roles, best_vals, best_layers


def best_in_distribution_layer_from_pairwise(
    probe_results: Mapping[str, Any],
    metric: str = "r2",
) -> int:
    """
    Single global best layer from in-distribution (diagonal) cross-role performance.

    Selects the layer with best mean diagonal metric across roles
    (maximize most metrics, minimize MSE).
    """
    pr = probe_results.get("pairwise_role_metrics")
    if not pr or "by_layer" not in pr:
        raise ValueError(
            "This cross_role JSON has no pairwise_role_metrics (re-run train after upgrading lp/train.py)."
        )
    roles = list(pr["roles"])
    by_layer = pr["by_layer"]
    layer_ids = sorted(int(k) for k in by_layer.keys())
    if not layer_ids:
        raise ValueError("pairwise_role_metrics.by_layer is empty")

    maximize = metric.lower() != "mse"
    best_layer: Optional[int] = None
    best_score: Optional[float] = None
    for L in layer_ids:
        blk = by_layer[str(L)]
        diag_vals: List[float] = []
        for r in roles:
            cell = blk.get(r, {}).get(r)
            if not cell:
                continue
            v = float(cell.get(metric, float("nan")))
            if np.isfinite(v):
                diag_vals.append(v)
        if not diag_vals:
            continue
        score = float(np.mean(diag_vals))
        if best_score is None or (score > best_score if maximize else score < best_score):
            best_score = score
            best_layer = L

    if best_layer is None:
        raise ValueError(f"No finite diagonal values found for metric={metric!r}")
    return int(best_layer)


def plot_cross_role_generalization_and_utility(
    cross_role_results_path: Path,
    *,
    repo_root: Path,
    model_key: str,
    options_rel: str,
    layer: Optional[int] = None,
    best_layer_per_pair: bool = False,
    best_layer_global_in_distribution: bool = False,
    gen_metric: str = "r2",
    similarity_metric: str = "correlation",
    role_display: Optional[Callable[[str], str]] = None,
    role_to_utilities_path: Optional[Mapping[str, Path | str]] = None,
) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    """
    Three figures matching the full-experiment notebook:

    1. Heatmap of pairwise probe generalization (row = train role, column = test role).
    2. Heatmap of utility-vector similarity between roles (same option ordering as collect).
    3. Scatter of off-diagonal pairs: utility similarity vs probe metric, with Pearson r.

    ``layer`` defaults to the best layer by mean leave-one-role-out ``gen_metric`` on the same JSON.
    If ``best_layer_per_pair=True``, each train/test pair uses its own best layer.
    If ``best_layer_global_in_distribution=True``, choose a single global layer by
    best mean in-distribution (diagonal) performance and use it for all pairs.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    cross_role_results_path = Path(cross_role_results_path)
    data = json.loads(cross_role_results_path.read_text())
    if data.get("probe_mode") != "cross_role":
        raise ValueError("Expected a cross_role probe_results JSON")

    if best_layer_per_pair and best_layer_global_in_distribution:
        raise ValueError("Choose only one: best_layer_per_pair or best_layer_global_in_distribution.")

    if best_layer_per_pair:
        roles, gen_mat, best_layer_mat = pairwise_metric_matrix_best_by_cell(data, gen_metric)
        layer_title = "best layer per role pair"
        layer_strategy = "best_per_pair"
    elif best_layer_global_in_distribution:
        layer = best_in_distribution_layer_from_pairwise(data, gen_metric)
        roles, gen_mat = pairwise_metric_matrix(data, layer, gen_metric)
        best_layer_mat = None
        layer_title = f"Layer {layer}"
        layer_strategy = "global_best_in_distribution"
        print(f"Using chosen layer: {layer} (global best in-distribution, metric={gen_metric})")
    else:
        if layer is None:
            summ = best_layers_summary(
                cross_role_results_path, primary_metric=gen_metric if gen_metric != "mse" else None
            )
            layer = int(summ["best_layer_primary"])
        roles, gen_mat = pairwise_metric_matrix(data, layer, gen_metric)
        best_layer_mat = None
        layer_title = f"layer {layer}"
        layer_strategy = "single_layer"
        print(f"Using chosen layer: {layer} (single-layer mode, metric={gen_metric})")
    if role_display:
        labels = [role_display(r) for r in roles]
    else:
        role_labels_by_key = {_canonicalize_role_key(k): v for k, v in _ROLE_LABELS.items()}
        missing_labels = [r for r in roles if _canonicalize_role_key(r) not in role_labels_by_key]
        if missing_labels:
            raise KeyError(
                "Missing role label mapping(s) for: "
                + ", ".join(sorted(missing_labels))
                + ". Add these to _ROLE_LABELS or pass role_display=..."
            )
        labels = [role_labels_by_key[_canonicalize_role_key(r)] for r in roles]

    opt_path = (linear_probes_dir(repo_root) / options_rel).resolve()
    _, util_mat = role_utility_matrix(
        repo_root,
        model_key,
        roles,
        opt_path,
        role_to_utilities_path=role_to_utilities_path,
    )
    sim_mat = utility_similarity_from_vectors(util_mat, similarity_metric)

    def _heatmap_with_annotations(
        ax: Any,
        mat: np.ndarray,
        row_labels: List[str],
        col_labels: List[str],
        *,
        title: str,
        cmap: str,
        vmin: float,
        vmax: float,
        xlabel: str,
        ylabel: str,
    ) -> None:
        n_r, n_c = mat.shape
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks(np.arange(n_c))
        ax.set_yticks(np.arange(n_r))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")
        ax.set_yticklabels(row_labels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig = ax.figure
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(np.arange(n_c), minor=True)
        ax.set_yticks(np.arange(n_r), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.25)
        ax.tick_params(which="minor", bottom=False, left=False)

    fig1, ax1 = plt.subplots(figsize=(max(7.0, 0.55 * len(roles)), max(5.5, 0.45 * len(roles))))
    fin = gen_mat[np.isfinite(gen_mat)]
    vmin_g = float(fin.min()) if fin.size else 0.0
    vmax_g = float(fin.max()) if fin.size else 1.0
    if vmin_g == vmax_g:
        vmax_g = vmin_g + 1e-6
    _heatmap_with_annotations(
        ax1,
        gen_mat,
        labels,
        labels,
        title=f"Probe Generalization Between Roles\n({layer_title})",
        cmap="viridis",
        vmin=vmin_g,
        vmax=vmax_g,
        xlabel="Test role",
        ylabel="Train role",
    )
    fig1.tight_layout()

    slabel = "Pearson r" if similarity_metric.lower() == "correlation" else "Cosine similarity"
    fig2, ax2 = plt.subplots(figsize=(max(7.0, 0.55 * len(roles)), max(5.5, 0.45 * len(roles))))
    _heatmap_with_annotations(
        ax2,
        sim_mat,
        labels,
        labels,
        title=f"{slabel} of per-role utility vectors\n(same options as collect)",
        cmap="RdYlGn",
        vmin=-1.0,
        vmax=1.0,
        xlabel="Role",
        ylabel="Role",
    )
    fig2.tight_layout()

    xs: List[float] = []
    ys: List[float] = []
    for i in range(len(roles)):
        for j in range(len(roles)):
            if i == j:
                continue
            xs.append(float(sim_mat[i, j]))
            ys.append(float(gen_mat[i, j]))
    x_arr = np.array(xs, dtype=np.float64)
    y_arr = np.array(ys, dtype=np.float64)
    ok = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[ok], y_arr[ok]

    fig3, ax3 = plt.subplots(figsize=(6.0, 5.0))
    ax3.scatter(x_arr, y_arr, alpha=0.85, edgecolors="k", linewidths=0.3)
    if x_arr.size >= 2 and np.std(x_arr) > 1e-12 and np.std(y_arr) > 1e-12:
        lr = stats.linregress(x_arr, y_arr)
        r2 = float(lr.rvalue**2)
        x_line = np.linspace(float(np.min(x_arr)), float(np.max(x_arr)), num=200)
        y_line = lr.intercept + lr.slope * x_line
        ax3.plot(x_line, y_line, color="tab:blue", linewidth=2.0, label="Linear fit")

        # 95% confidence band for the fitted mean response.
        n = int(x_arr.size)
        if n > 2:
            y_hat = lr.intercept + lr.slope * x_arr
            residuals = y_arr - y_hat
            sxx = float(np.sum((x_arr - np.mean(x_arr)) ** 2))
            if sxx > 1e-12:
                s_err = float(np.sqrt(np.sum(residuals**2) / (n - 2)))
                se_fit = s_err * np.sqrt((1.0 / n) + ((x_line - np.mean(x_arr)) ** 2) / sxx)
                t_crit = float(stats.t.ppf(0.975, n - 2))
                ci_delta = t_crit * se_fit
                ax3.fill_between(
                    x_line,
                    y_line - ci_delta,
                    y_line + ci_delta,
                    color="tab:blue",
                    alpha=0.18,
                    linewidth=0.0,
                    label="95% CI",
                )
        ax3.set_title(f"Cross-Role Performance vs. Utility Similarity\nLinear fit $R^2 = {r2:.3f}$")
    else:
        ax3.set_title("Off-diagonal role pairs")

    similarity_label = (
        "Utility similarity (Pearson correlation)"
        if similarity_metric.lower() == "correlation"
        else "Utility similarity (cosine similarity)"
    )
    gen_metric_label = {
        "r2": "R^2",
        "mse": "MSE",
        "mae": "MAE",
    }.get(gen_metric.lower(), gen_metric.upper())
    ax3.set_xlabel(similarity_label)
    ax3.set_ylabel(f"Cross-role generalization accuracy")
    ax3.legend(loc="best", frameon=False)
    ax3.grid(alpha=0.25)
    fig3.tight_layout()

    info: Dict[str, Any] = {
        "layer": layer,
        "best_layer_per_pair": bool(best_layer_per_pair),
        "best_layer_global_in_distribution": bool(best_layer_global_in_distribution),
        "layer_strategy": layer_strategy,
        "roles": roles,
        "n_offdiag_pairs": int(x_arr.size),
    }
    if best_layer_mat is not None:
        info["best_layer_by_pair"] = best_layer_mat
    return fig1, fig2, fig3, info


def compute_cross_role_activation_similarity_results(
    cross_role_results_path: Path,
    *,
    layer: int,
    output_path: Optional[Path | str] = None,
    metadata_path: Optional[Path | str] = None,
    layers_path: Optional[Path | str] = None,
    activations_path: Optional[Path | str] = None,
) -> Path:
    """Compute and save cross-role activation/utility similarity matrices for one layer."""
    import torch

    cross_role_results_path = Path(cross_role_results_path)
    data = json.loads(cross_role_results_path.read_text())
    if data.get("probe_mode") != "cross_role":
        raise ValueError("Expected a cross_role probe_results JSON")

    save_suffix = str(data.get("save_suffix", "")).strip()
    if not save_suffix:
        raise ValueError("cross_role results JSON missing save_suffix")
    position = str(data.get("position", "gen_first"))
    prefix = cross_role_results_path.parent / f"linear_probes_{save_suffix}"
    out_path = (
        Path(output_path)
        if output_path is not None
        else cross_role_results_path.parent / f"linear_probes_{save_suffix}_activation_similarity_layer_{int(layer)}.json"
    )

    meta_p = Path(metadata_path) if metadata_path is not None else Path(str(prefix) + "_metadata.jsonl")
    layers_p = Path(layers_path) if layers_path is not None else Path(str(prefix) + "_layers.json")
    acts_p = (
        Path(activations_path)
        if activations_path is not None
        else Path(str(prefix) + ("_X_gen_first.pt" if position == "gen_first" else "_X_prompt_last.pt"))
    )
    for p in (meta_p, layers_p, acts_p):
        if not p.is_file():
            raise FileNotFoundError(f"Required artifact not found: {p}")

    with layers_p.open("r") as f:
        layer_info = json.load(f)
    layers = [int(x) for x in layer_info["layers"]]
    if int(layer) not in layers:
        raise KeyError(f"Layer {layer} not present in {layers_p}")
    layer_idx = layers.index(int(layer))
    print(f"Using chosen layer: {layer} (activation-similarity compute)")

    pack = torch.load(acts_p, map_location="cpu")
    X = pack.get("X")
    if X is None:
        raise KeyError(f"Expected key 'X' in activations pack: {acts_p}")
    X_np = X.numpy() if hasattr(X, "numpy") else np.asarray(X)
    if X_np.ndim != 3:
        raise ValueError(f"Expected X with shape (N, L, D), got {tuple(X_np.shape)}")
    X_layer = np.asarray(X_np[:, layer_idx, :], dtype=np.float64)

    metas: List[Dict[str, Any]] = []
    with meta_p.open("r") as f:
        for line in f:
            metas.append(json.loads(line))
    if len(metas) != X_layer.shape[0]:
        raise ValueError(f"Metadata/activation size mismatch: {len(metas)} vs {X_layer.shape[0]}")

    role_option_sum: Dict[str, Dict[str, np.ndarray]] = {}
    role_option_count: Dict[str, Dict[str, int]] = {}
    role_utility_sum: Dict[str, Dict[str, float]] = {}
    role_utility_count: Dict[str, Dict[str, int]] = {}
    for idx, m in enumerate(metas):
        role = str(m["role"])
        oid = str(m["option_id"])
        role_option_sum.setdefault(role, {})
        role_option_count.setdefault(role, {})
        role_utility_sum.setdefault(role, {})
        role_utility_count.setdefault(role, {})
        if oid not in role_option_sum[role]:
            role_option_sum[role][oid] = np.array(X_layer[idx], dtype=np.float64, copy=True)
            role_option_count[role][oid] = 1
            role_utility_sum[role][oid] = float(m["utility"])
            role_utility_count[role][oid] = 1
        else:
            role_option_sum[role][oid] += X_layer[idx]
            role_option_count[role][oid] += 1
            role_utility_sum[role][oid] += float(m["utility"])
            role_utility_count[role][oid] += 1

    role_option_vec: Dict[str, Dict[str, np.ndarray]] = {}
    role_utility_vec: Dict[str, Dict[str, float]] = {}
    for role, by_option in role_option_sum.items():
        role_option_vec[role] = {}
        role_utility_vec[role] = {}
        for oid, vec_sum in by_option.items():
            v = vec_sum / max(1, role_option_count[role][oid])
            nrm = float(np.linalg.norm(v))
            role_option_vec[role][oid] = (v / nrm) if nrm > 0 else v
            role_utility_vec[role][oid] = role_utility_sum[role][oid] / max(1, role_utility_count[role][oid])

    roles = list(data["pairwise_role_metrics"]["roles"])
    n = len(roles)
    act_sim_mat = np.full((n, n), np.nan, dtype=np.float64)
    util_sim_mat = np.full((n, n), np.nan, dtype=np.float64)
    shared_counts = np.zeros((n, n), dtype=np.int32)
    for i, ri in enumerate(roles):
        opts_i = role_option_vec.get(ri, {})
        u_i = role_utility_vec.get(ri, {})
        for j, rj in enumerate(roles):
            opts_j = role_option_vec.get(rj, {})
            u_j = role_utility_vec.get(rj, {})
            shared = sorted(set(opts_i.keys()).intersection(opts_j.keys()))
            if not shared:
                continue
            cos_vals: List[float] = []
            ui: List[float] = []
            uj: List[float] = []
            for oid in shared:
                c = float(np.dot(opts_i[oid], opts_j[oid]))
                if np.isfinite(c):
                    cos_vals.append(c)
                if oid in u_i and oid in u_j:
                    ui.append(float(u_i[oid]))
                    uj.append(float(u_j[oid]))
            if cos_vals:
                act_sim_mat[i, j] = float(np.mean(cos_vals))
                shared_counts[i, j] = int(len(cos_vals))
            if len(ui) >= 2 and np.std(ui) > 1e-12 and np.std(uj) > 1e-12:
                util_sim_mat[i, j] = float(np.corrcoef(np.array(ui), np.array(uj))[0, 1])

    payload = {
        "cross_role_results_path": str(cross_role_results_path),
        "save_suffix": save_suffix,
        "position": position,
        "layer": int(layer),
        "roles": roles,
        "activation_similarity_matrix": act_sim_mat.tolist(),
        "utility_similarity_matrix": util_sim_mat.tolist(),
        "shared_option_counts": shared_counts.astype(int).tolist(),
        "metadata_path": str(meta_p),
        "layers_path": str(layers_p),
        "activations_path": str(acts_p),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote activation similarity results: {out_path}")
    return out_path


def plot_cross_role_generalization_and_activation_similarity_from_results(
    cross_role_results_path: Path,
    activation_similarity_results_path: Path | str,
    *,
    gen_metric: str = "r2",
    role_display: Optional[Callable[[str], str]] = None,
) -> Tuple[Any, Any, Any, Any, Dict[str, Any]]:
    """Plot probe generalization vs saved activation/utility similarity results."""
    import matplotlib.pyplot as plt
    from scipy import stats

    cross_role_results_path = Path(cross_role_results_path)
    data = json.loads(cross_role_results_path.read_text())
    if data.get("probe_mode") != "cross_role":
        raise ValueError("Expected a cross_role probe_results JSON")

    res = json.loads(Path(activation_similarity_results_path).read_text())
    layer = int(res["layer"])
    roles = list(res["roles"])
    act_sim_mat = np.asarray(res["activation_similarity_matrix"], dtype=np.float64)
    util_sim_mat = np.asarray(res["utility_similarity_matrix"], dtype=np.float64)
    shared_counts = np.asarray(res["shared_option_counts"], dtype=np.int32)

    roles_gen, gen_mat = pairwise_metric_matrix(data, layer, gen_metric)
    if roles_gen != roles:
        raise ValueError("Role ordering mismatch between cross-role results and activation-similarity results")
    if role_display:
        labels = [role_display(r) for r in roles]
    else:
        role_labels_by_key = {_canonicalize_role_key(k): v for k, v in _ROLE_LABELS.items()}
        missing_labels = [r for r in roles if _canonicalize_role_key(r) not in role_labels_by_key]
        if missing_labels:
            raise KeyError(
                "Missing role label mapping(s) for: "
                + ", ".join(sorted(missing_labels))
                + ". Add these to _ROLE_LABELS or pass role_display=..."
            )
        labels = [role_labels_by_key[_canonicalize_role_key(r)] for r in roles]

    n = len(roles)

    def _heatmap(ax: Any, mat: np.ndarray, *, title: str, cmap: str, vmin: float, vmax: float, xlabel: str, ylabel: str) -> None:
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(np.arange(n), minor=True)
        ax.set_yticks(np.arange(n), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.25)
        ax.tick_params(which="minor", bottom=False, left=False)

    fig1, ax1 = plt.subplots(figsize=(max(7.0, 0.55 * n), max(5.5, 0.45 * n)))
    fin = gen_mat[np.isfinite(gen_mat)]
    vmin_g = float(fin.min()) if fin.size else 0.0
    vmax_g = float(fin.max()) if fin.size else 1.0
    if vmin_g == vmax_g:
        vmax_g = vmin_g + 1e-6
    _heatmap(ax1, gen_mat, title=f"Cross-role probe generalization ({gen_metric}) @ layer {layer}", cmap="viridis", vmin=vmin_g, vmax=vmax_g, xlabel="Test role", ylabel="Train role")
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(max(7.0, 0.55 * n), max(5.5, 0.45 * n)))
    _heatmap(ax2, act_sim_mat, title=f"Average matched-outcome activation cosine similarity @ layer {layer}", cmap="RdYlGn", vmin=-1.0, vmax=1.0, xlabel="Test role", ylabel="Train role")
    fig2.tight_layout()

    xs: List[float] = []
    ys: List[float] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            xs.append(float(act_sim_mat[i, j]))
            ys.append(float(gen_mat[i, j]))
    x_arr = np.array(xs, dtype=np.float64)
    y_arr = np.array(ys, dtype=np.float64)
    ok = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[ok], y_arr[ok]

    def _scatter_with_fit(ax: Any, x: np.ndarray, y: np.ndarray, *, title_prefix: str, xlabel: str, ylabel: str) -> float:
        ax.scatter(x, y, alpha=0.85, edgecolors="k", linewidths=0.3)
        r2 = float("nan")
        if x.size >= 2 and np.std(x) > 1e-12 and np.std(y) > 1e-12:
            lr = stats.linregress(x, y)
            r2 = float(lr.rvalue**2)
            x_line = np.linspace(float(np.min(x)), float(np.max(x)), num=200)
            y_line = lr.intercept + lr.slope * x_line
            ax.plot(x_line, y_line, color="tab:blue", linewidth=2.0, label="Linear fit")
            if x.size > 2:
                y_hat = lr.intercept + lr.slope * x
                residuals = y - y_hat
                sxx = float(np.sum((x - np.mean(x)) ** 2))
                if sxx > 1e-12:
                    s_err = float(np.sqrt(np.sum(residuals**2) / (x.size - 2)))
                    se_fit = s_err * np.sqrt((1.0 / x.size) + ((x_line - np.mean(x)) ** 2) / sxx)
                    t_crit = float(stats.t.ppf(0.975, x.size - 2))
                    d = t_crit * se_fit
                    ax.fill_between(x_line, y_line - d, y_line + d, color="tab:blue", alpha=0.18, label="95% CI")
            ax.set_title(f"{title_prefix}\nLinear fit $R^2$ = {r2:.3f}")
        else:
            ax.set_title(title_prefix)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", frameon=False)
        ax.grid(alpha=0.25)
        return r2

    fig3, ax3 = plt.subplots(figsize=(6.0, 5.0))
    r2_gen_act = _scatter_with_fit(
        ax3,
        x_arr,
        y_arr,
        title_prefix="Cross-role performance vs activation similarity",
        xlabel="Activation similarity (average matched-outcome cosine)",
        ylabel="Cross-role generalization",
    )
    fig3.tight_layout()

    xu: List[float] = []
    yu: List[float] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            xu.append(float(act_sim_mat[i, j]))
            yu.append(float(util_sim_mat[i, j]))
    xu_arr = np.array(xu, dtype=np.float64)
    yu_arr = np.array(yu, dtype=np.float64)
    oku = np.isfinite(xu_arr) & np.isfinite(yu_arr)
    xu_arr, yu_arr = xu_arr[oku], yu_arr[oku]

    fig4, ax4 = plt.subplots(figsize=(6.0, 5.0))
    r2_act_util = _scatter_with_fit(
        ax4,
        xu_arr,
        yu_arr,
        title_prefix="Activation similarity vs utility similarity",
        xlabel="Activation similarity (average matched-outcome cosine)",
        ylabel="Utility similarity (Pearson correlation)",
    )
    fig4.tight_layout()

    info: Dict[str, Any] = {
        "layer": int(layer),
        "roles": roles,
        "n_offdiag_pairs_gen_vs_act": int(x_arr.size),
        "n_offdiag_pairs_act_vs_util": int(xu_arr.size),
        "r2_gen_vs_act": r2_gen_act,
        "r2_act_vs_util": r2_act_util,
        "shared_option_counts": shared_counts,
    }
    return fig1, fig2, fig3, fig4, info


def plot_cross_role_generalization_and_activation_similarity(
    cross_role_results_path: Path,
    *,
    layer: int,
    gen_metric: str = "r2",
    role_display: Optional[Callable[[str], str]] = None,
    metadata_path: Optional[Path | str] = None,
    layers_path: Optional[Path | str] = None,
    activations_path: Optional[Path | str] = None,
    activation_similarity_results_path: Optional[Path | str] = None,
) -> Tuple[Any, Any, Any, Any, Dict[str, Any]]:
    """
    Convenience wrapper: compute cached activation-similarity results, then plot from that file.
    """
    out_path = compute_cross_role_activation_similarity_results(
        cross_role_results_path,
        layer=layer,
        output_path=activation_similarity_results_path,
        metadata_path=metadata_path,
        layers_path=layers_path,
        activations_path=activations_path,
    )
    return plot_cross_role_generalization_and_activation_similarity_from_results(
        cross_role_results_path,
        out_path,
        gen_metric=gen_metric,
        role_display=role_display,
    )
