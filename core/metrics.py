"""Forecast evaluation metrics (computed on the original scale)."""
from __future__ import annotations

import numpy as np

METRIC_INFO = {
    "MAE": "平均绝对误差",
    "RMSE": "均方根误差",
    "MAPE": "平均绝对百分比误差 (%)，排除真实值 < 全局均值 1% 的点",
    "sMAPE": "对称平均绝对百分比误差 (%)",
    "WAPE": "加权绝对百分比误差 (%) = Σ|e| / Σ|y|",
    "R2": "决定系数",
}


def _core(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    e = p - y
    ae = np.abs(e)
    mask = np.abs(y) > thr
    ss_res = (e ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return {
        "MAE": float(ae.mean()),
        "RMSE": float(np.sqrt((e ** 2).mean())),
        "MAPE": float((ae[mask] / np.abs(y[mask])).mean() * 100) if mask.any() else None,
        "sMAPE": float((2 * ae / np.maximum(np.abs(y) + np.abs(p), 1e-9)).mean() * 100),
        "WAPE": float(ae.sum() / max(np.abs(y).sum(), 1e-9) * 100),
        "R2": float(1 - ss_res / ss_tot) if ss_tot > 0 else None,
    }


def evaluate(y: np.ndarray, p: np.ndarray) -> dict:
    """y, p: (n, H, N) on original scale."""
    thr = 0.01 * float(np.abs(y).mean())
    H, N = y.shape[1], y.shape[2]
    overall = _core(y, p, thr)
    per_h = [_core(y[:, h], p[:, h], thr) for h in range(H)]
    per_n = [_core(y[:, :, n], p[:, :, n], thr) for n in range(N)]
    keys = list(overall)
    return {
        "overall": overall,
        "per_horizon": {k: [d[k] for d in per_h] for k in keys},
        "per_region": {k: [d[k] for d in per_n] for k in keys},
    }
