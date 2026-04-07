from __future__ import annotations

import numpy as np


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            avg = ranks[order[i : j + 1]].mean()
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata(x.astype(np.float64))
    ry = rankdata(y.astype(np.float64))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx**2).sum()) * np.sqrt((ry**2).sum())
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def ridge_fit_closed_form(X: np.ndarray, y: np.ndarray, ridge_lambda: float) -> tuple[np.ndarray, float]:
    X = X.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    x_mean = X.mean(axis=0, keepdims=True)
    y_mean = y.mean()
    Xc = X - x_mean
    yc = y - y_mean
    d = X.shape[1]
    A = Xc.T @ Xc + ridge_lambda * np.eye(d, dtype=np.float64)
    bvec = Xc.T @ yc
    w = np.linalg.solve(A, bvec)
    b0 = float(y_mean - (x_mean @ w).item())
    return w.astype(np.float32), b0


def ridge_predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return X @ w + b
