from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from lp.metrics import (
    pairwise_preference_accuracy,
    r2_score,
    ridge_fit_closed_form,
    ridge_predict,
    spearmanr,
)


def train(args: argparse.Namespace) -> None:
    os.makedirs(args.save_dir, exist_ok=True)
    save_suffix = args.save_suffix or args.model_key
    out_prefix = os.path.join(args.save_dir, f"linear_probes_{save_suffix}")

    meta_path = out_prefix + "_metadata.jsonl"
    layers_path = out_prefix + "_layers.json"
    pos_to_file = {
        "gen_first": out_prefix + "_X_gen_first.pt",
        "prompt_last": out_prefix + "_X_prompt_last.pt",
        "option_a_last": out_prefix + "_X_option_a_last.pt",
        "option_b_last": out_prefix + "_X_option_b_last.pt",
    }
    if args.position not in pos_to_file:
        raise ValueError(f"Unknown position: {args.position}")
    X_path = pos_to_file[args.position]

    metas: List[Dict[str, Any]] = []
    with open(meta_path, "r") as f:
        for line in f:
            d = json.loads(line)
            metas.append(d)

    with open(layers_path, "r") as f:
        layer_info = json.load(f)
    layers: List[int] = list(layer_info["layers"])

    pack = torch.load(X_path, map_location="cpu")
    X = pack["X"]

    target_key = {
        "utility": "utility",
        "rating": "rating",
        "utility_a": "utility_a",
        "utility_b": "utility_b",
    }.get(args.target)
    if target_key is None:
        raise ValueError(f"Unknown target: {args.target}")

    keep = np.array([m.get(target_key) is not None for m in metas], dtype=bool)
    kept_indices = np.where(keep)[0]
    if len(kept_indices) < 10:
        raise ValueError(f"Too few valid examples after filtering: {len(kept_indices)}")

    X = X[kept_indices]
    kept_metas = [metas[i] for i in kept_indices.tolist()]

    y = np.array([float(m[target_key]) for m in kept_metas], dtype=np.float32)
    roles = np.array([str(m.get("role", "")) for m in kept_metas], dtype=object)

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
            pacc = pairwise_preference_accuracy(yte, yhat)
            layer_metrics: Dict[str, float] = {
                "mse": mse,
                "r2": float(r2),
                "spearman": float(spr),
                "pairwise_pref_acc": float(pacc),
            }
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

        min_ex = 10

        def _metrics_train_test(train_idx: np.ndarray, test_idx: np.ndarray, li: int) -> Dict[str, float]:
            X_np = X.numpy().astype(np.float32, copy=False)
            Xtr = X_np[train_idx, li, :]
            Xte = X_np[test_idx, li, :]
            ytr = y[train_idx]
            yte = y[test_idx]
            w, b0 = ridge_fit_closed_form(Xtr, ytr, ridge_lambda=args.ridge_lambda)
            yhat = ridge_predict(Xte, w, b0)
            mse = float(((yte - yhat) ** 2).mean())
            r2 = r2_score(yte, yhat)
            spr = spearmanr(yte, yhat)
            pacc = pairwise_preference_accuracy(yte, yhat)
            out: Dict[str, float] = {
                "mse": mse,
                "r2": float(r2),
                "spearman": float(spr),
                "pairwise_pref_acc": float(pacc),
            }
            if args.target == "rating":
                yhat_int = np.clip(np.rint(yhat), 1, 10).astype(np.int32)
                yte_int = np.clip(np.rint(yte), 1, 10).astype(np.int32)
                out["accuracy"] = float((yhat_int == yte_int).mean())
            return out

        nan_block = {k: float("nan") for k in ("mse", "r2", "spearman", "pairwise_pref_acc")}
        if args.target == "rating":
            nan_block["accuracy"] = float("nan")

        pairwise_by_layer: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
        for li, layer in enumerate(layers):
            lk = str(layer)
            pairwise_by_layer[lk] = {}
            for tr in all_roles:
                pairwise_by_layer[lk][tr] = {}
                for te in all_roles:
                    if tr == te:
                        ridx = np.where(roles == te)[0]
                        if len(ridx) < min_ex:
                            pairwise_by_layer[lk][tr][te] = dict(nan_block)
                            continue
                        tr_idx, te_idx = split_indices(ridx)
                        if len(tr_idx) < min_ex or len(te_idx) < min_ex:
                            pairwise_by_layer[lk][tr][te] = dict(nan_block)
                            continue
                        pairwise_by_layer[lk][tr][te] = _metrics_train_test(tr_idx, te_idx, li)
                    else:
                        tr_idx = np.where(roles == tr)[0]
                        te_idx = np.where(roles == te)[0]
                        if len(tr_idx) < min_ex or len(te_idx) < min_ex:
                            pairwise_by_layer[lk][tr][te] = dict(nan_block)
                            continue
                        pairwise_by_layer[lk][tr][te] = _metrics_train_test(tr_idx, te_idx, li)

        results["pairwise_role_metrics"] = {
            "roles": all_roles,
            "description": (
                "Rows = train role, columns = test role. Diagonal: ridge probe trained on a random "
                "within-role train split, evaluated on the held-out split for that role. Off-diagonal: "
                "trained on all examples from the row role, evaluated on all examples from the column role."
            ),
            "by_layer": pairwise_by_layer,
        }
    else:
        raise ValueError(f"Unknown probe_mode: {args.probe_mode}")

    out_path = out_prefix + f"_probe_results_{args.position}_{args.target}_{args.probe_mode}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[train] wrote probe metrics to {out_path}")
