from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from lp.data import ExampleMeta
from lp.metrics import r2_score, ridge_fit_closed_form, ridge_predict, spearmanr


def train(args: argparse.Namespace) -> None:
    os.makedirs(args.save_dir, exist_ok=True)
    save_suffix = args.save_suffix or args.model_key
    out_prefix = os.path.join(args.save_dir, f"linear_probes_{save_suffix}")

    meta_path = out_prefix + "_metadata.jsonl"
    layers_path = out_prefix + "_layers.json"
    X_path = out_prefix + ("_X_gen_first.pt" if args.position == "gen_first" else "_X_prompt_last.pt")

    metas: List[ExampleMeta] = []
    with open(meta_path, "r") as f:
        for line in f:
            d = json.loads(line)
            metas.append(
                ExampleMeta(
                    role=d["role"],
                    option_id=str(d["option_id"]),
                    rating=d.get("rating"),
                    utility=float(d["utility"]),
                )
            )

    with open(layers_path, "r") as f:
        layer_info = json.load(f)
    layers: List[int] = list(layer_info["layers"])

    pack = torch.load(X_path, map_location="cpu")
    X = pack["X"]

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

    def eval_one(train_idx: np.ndarray, test_idx: np.ndarray) -> Dict[int, Dict[str, float]]:
        metrics_by_layer: Dict[int, Dict[str, float]] = {}
        X_np = X.numpy().astype(np.float32, copy=False)
        for li, layer in enumerate(layers):
            Xtr = X_np[train_idx, li, :]
            Xte = X_np[test_idx, li, :]
            ytr = y[train_idx]
            yte = y[test_idx]
            w, b0 = ridge_fit_closed_form(Xtr, ytr, ridge_lambda=args.ridge_lambda)
            yhat = ridge_predict(Xte, w, b0)
            mse = float(((yte - yhat) ** 2).mean())
            r2 = r2_score(yte, yhat)
            spr = spearmanr(yte, yhat)
            layer_metrics: Dict[str, float] = {"mse": mse, "r2": float(r2), "spearman": float(spr)}
            if args.target == "rating":
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
