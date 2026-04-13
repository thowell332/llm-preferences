"""
Helpers for ``linear_probes_experiment.ipynb`` (Google Colab or local).

Runs ``run_linear_probes.py`` as a subprocess (same cwd/relative paths as the CLI)
and provides small plotting utilities so notebooks stay thin.
"""
from __future__ import annotations

import json
import os
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
    "probe_results_path",
    "existing_probe_results_path",
    "default_metric_name",
    "best_layers_summary",
    "plot_probe_results_file",
    "plot_cross_role_generalization_and_utility",
    "role_utility_matrix",
    "utility_similarity_from_vectors",
    "pairwise_metric_matrix",
    "artifact_paths",
]


def linear_probes_dir(repo_root: Path) -> Path:
    return Path(repo_root).resolve() / "utility_analysis" / "experiments" / "linear_probes"


def default_utilities_rel(model_key: str, role: str) -> str:
    role_slug = role.replace(" ", "_")
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
    rc = subprocess.run(cmd, cwd=str(lp), env=env)
    if rc.returncode != 0:
        raise RuntimeError(f"run_linear_probes exited with code {rc.returncode}")


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
        "layers_json": Path(str(pfx) + "_layers.json"),
        "x_gen_first_pt": Path(str(pfx) + "_X_gen_first.pt"),
        "x_prompt_last_pt": Path(str(pfx) + "_X_prompt_last.pt"),
        "run_metadata_json": Path(str(pfx) + "_run_metadata.json"),
    }


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

    cargv = _collect_argv(
        model_key=model_key,
        save_dir=save_dir,
        save_suffix=save_suffix,
        options_path=options_path,
        utilities_path=utilities_path,
        utilities_dir=utilities_dir,
        roles=roles,
        roleset=roleset,
        roles_config_path=roles_config_path,
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


def plot_cross_role_generalization_and_utility(
    cross_role_results_path: Path,
    *,
    repo_root: Path,
    model_key: str,
    options_rel: str,
    layer: Optional[int] = None,
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
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    cross_role_results_path = Path(cross_role_results_path)
    data = json.loads(cross_role_results_path.read_text())
    if data.get("probe_mode") != "cross_role":
        raise ValueError("Expected a cross_role probe_results JSON")

    if layer is None:
        summ = best_layers_summary(cross_role_results_path, primary_metric=gen_metric if gen_metric != "mse" else None)
        layer = int(summ["best_layer_primary"])

    roles, gen_mat = pairwise_metric_matrix(data, layer, gen_metric)
    labels = [role_display(r) if role_display else r for r in roles]

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
        for i in range(n_r):
            for j in range(n_c):
                v = mat[i, j]
                if not np.isfinite(v):
                    t = ""
                else:
                    t = f"{v:.2f}"
                ax.text(j, i, t, ha="center", va="center", color="0.05", fontsize=8)
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
        title=f"Pairwise {gen_metric} @ layer {layer}\n(train row → test column)",
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
        r_p, p_v = stats.pearsonr(x_arr, y_arr)
        ax3.set_title(f"Off-diagonal role pairs (n={x_arr.size})\nPearson r = {r_p:.3f}, p = {p_v:.3g}")
    else:
        ax3.set_title(f"Off-diagonal role pairs (n={x_arr.size})")
    ax3.set_xlabel(f"Utility-vector {similarity_metric}")
    ax3.set_ylabel(f"Probe {gen_metric} (train → test)")
    ax3.grid(alpha=0.25)
    fig3.tight_layout()

    info: Dict[str, Any] = {
        "layer": layer,
        "roles": roles,
        "n_offdiag_pairs": int(x_arr.size),
    }
    return fig1, fig2, fig3, info
