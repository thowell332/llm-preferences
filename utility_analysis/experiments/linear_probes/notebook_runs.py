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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
    utilities_path: str,
    roles: str,
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
        "--utilities_path",
        utilities_path,
        "--roles",
        roles,
        "--layers",
        layers,
        "--max_new_tokens_for_parsing",
        str(max_new_tokens_for_parsing),
        "--max_model_len",
        str(max_model_len),
        "--max_examples",
        str(max_examples),
    ]
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
    utilities_path: str,
    roles: str,
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
    extra_env: Dict[str, str] = {}
    if backend == "hf":
        extra_env["CUDA_LAUNCH_BLOCKING"] = "1"

    cargv = _collect_argv(
        model_key=model_key,
        save_dir=save_dir,
        save_suffix=save_suffix,
        options_path=options_path,
        utilities_path=utilities_path,
        roles=roles,
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
