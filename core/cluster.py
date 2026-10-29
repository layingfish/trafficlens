"""Clustering regions by traffic patterns to discover functional zones."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import cluster as skc
from sklearn import metrics as skm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .data import Dataset
from .eda import _r, hourly, is_offday, region_stats

METHODS = {
    "kmeans": {"label": "K-Means", "params": [
        {"name": "k", "label": "簇数 k", "type": "int", "default": 5, "min": 2, "max": 8},
        {"name": "n_init", "label": "初始化次数", "type": "int", "default": 10, "min": 1, "max": 50}]},
    "agglomerative": {"label": "层次聚类 Agglomerative", "params": [
        {"name": "k", "label": "簇数 k", "type": "int", "default": 5, "min": 2, "max": 8},
        {"name": "linkage", "label": "连接方式", "type": "select", "default": "ward",
         "options": ["ward", "complete", "average", "single"]}]},
    "gmm": {"label": "高斯混合 GMM", "params": [
        {"name": "k", "label": "成分数 k", "type": "int", "default": 5, "min": 2, "max": 8},
        {"name": "covariance_type", "label": "协方差类型", "type": "select", "default": "diag",
         "options": ["full", "tied", "diag", "spherical"]}]},
    "spectral": {"label": "谱聚类 Spectral", "params": [
        {"name": "k", "label": "簇数 k", "type": "int", "default": 5, "min": 2, "max": 8},
        {"name": "n_neighbors", "label": "近邻数", "type": "int", "default": 10, "min": 3, "max": 50}]},
    "dbscan": {"label": "DBSCAN", "params": [
        {"name": "eps", "label": "邻域半径 eps", "type": "float", "default": 0.35, "min": 0.05, "max": 50, "step": 0.05},
        {"name": "min_samples", "label": "最小样本数", "type": "int", "default": 3, "min": 2, "max": 30}]},
    "hdbscan": {"label": "HDBSCAN", "params": [
        {"name": "min_cluster_size", "label": "最小簇大小", "type": "int", "default": 5, "min": 2, "max": 50}]},
}

FEATURES = {
    "wd_we": "工作日+休息日 日内曲线（48 维）",
    "daily": "平均日内曲线（24 维）",
    "weekly": "平均周内曲线（168 维）",
    "stats": "统计特征（均值/变异/峰时/昼夜比/周末比）",
}
NORMS = {
    "shape": "形状归一（除以区域均值，只看模式）",
    "zscore": "Z-score（每区域标准化）",
    "log": "log1p 原值（保留量级）",
}

ZONES = {
    "office": ("商务办公区", "工作日白天高峰明显、休息日流量显著下降"),
    "residential": ("居住区", "晚间/夜间占比高、工作日与休息日差异小"),
    "leisure": ("休闲商业/旅游区", "休息日流量相对上升，午后至晚间活跃"),
    "lowactivity": ("低活跃/郊野区", "流量量级低、日内波动平缓"),
    "mixed": ("混合功能区", "各项指标接近整体平均水平"),
}


def _profiles(ds: Dataset):
    h = hourly(ds)
    off = is_offday(ds, h.index)
    hod = h.index.hour
    wd = h[~off].groupby(hod[~off]).mean().reindex(range(24))
    we = h[off].groupby(hod[off]).mean().reindex(range(24))
    wk = h.groupby(h.index.dayofweek * 24 + hod).mean().reindex(range(168))
    return h, wd.T, we.T, wk.T  # each (N, 24|168)


def build_features(ds: Dataset, feature: str, norm: str) -> np.ndarray:
    h, wd, we, wk = _profiles(ds)
    if feature == "stats":
        st = region_stats(ds)
        X = np.c_[
            np.log1p(st["mean"]), st["cv"], np.sin(2 * np.pi * st["peak_hour"] / 24),
            np.cos(2 * np.pi * st["peak_hour"] / 24), st["offday_ratio"], st["night_share"],
            np.log1p(st["day_night_ratio"]),
        ]
        return StandardScaler().fit_transform(X)
    X = {"wd_we": np.c_[wd.values, we.values], "daily": (wd.values * 5 + we.values * 2) / 7, "weekly": wk.values}[feature]
    X = np.nan_to_num(X.astype(float))
    if norm == "shape":
        X = X / np.maximum(X.mean(axis=1, keepdims=True), 1e-9)
    elif norm == "zscore":
        X = (X - X.mean(axis=1, keepdims=True)) / np.maximum(X.std(axis=1, keepdims=True), 1e-9)
    else:
        X = np.log1p(X)
        X = X - X.mean()
    return X


def _fit(method: str, X: np.ndarray, p: dict, seed: int = 0) -> np.ndarray:
    k = int(p.get("k", 5))
    if method == "kmeans":
        return skc.KMeans(k, n_init=int(p.get("n_init", 10)), random_state=seed).fit_predict(X)
    if method == "agglomerative":
        return skc.AgglomerativeClustering(k, linkage=p.get("linkage", "ward")).fit_predict(X)
    if method == "gmm":
        return GaussianMixture(k, covariance_type=p.get("covariance_type", "diag"), random_state=seed,
                               reg_covar=1e-4).fit_predict(X)
    if method == "spectral":
        nn = min(int(p.get("n_neighbors", 10)), len(X) - 1)
        return skc.SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=nn,
                                      random_state=seed, assign_labels="cluster_qr").fit_predict(X)
    if method == "dbscan":
        return skc.DBSCAN(eps=float(p.get("eps", 0.35)), min_samples=int(p.get("min_samples", 3))).fit_predict(X)
    if method == "hdbscan":
        return skc.HDBSCAN(min_cluster_size=int(p.get("min_cluster_size", 5))).fit_predict(X)
    raise ValueError(f"unknown method {method}")


def _scores(X, labels):
    mask = labels >= 0
    uniq = np.unique(labels[mask])
    if len(uniq) < 2 or mask.sum() <= len(uniq):
        return {"silhouette": None, "calinski_harabasz": None, "davies_bouldin": None}
    return {
        "silhouette": float(skm.silhouette_score(X[mask], labels[mask])),
        "calinski_harabasz": float(skm.calinski_harabasz_score(X[mask], labels[mask])),
        "davies_bouldin": float(skm.davies_bouldin_score(X[mask], labels[mask])),
    }


def interpret(ds: Dataset, labels: np.ndarray) -> list:
    """Heuristic functional-zone naming from cluster-average indicators (relative to other clusters)."""
    h, wd, we, _ = _profiles(ds)
    st = region_stats(ds)
    rows = []
    for c in sorted(set(labels)):
        m = labels == c
        wdp, wep = wd.values[m].mean(0), we.values[m].mean(0)
        base = max(wdp.mean(), 1e-9)
        rows.append({
            "cluster": int(c),
            "size": int(m.sum()),
            "level": float(np.log1p(st["mean"].values[m]).mean()),
            "work": float(wdp[9:18].mean() / base),
            "evening": float(wdp[19:24].mean() / base),
            "night": float(wdp[0:6].mean() / base),
            "offday_ratio": float(wep.mean() / base),
            "amplitude": float((wdp.max() - wdp.min()) / base),
            "peak_hour_workday": int(np.argmax(wdp)),
            "peak_hour_offday": int(np.argmax(wep)),
        })
    df = pd.DataFrame(rows).set_index("cluster")
    z = (df - df.mean()) / df.std(ddof=0).replace(0, 1)
    z = z.fillna(0)
    scores = pd.DataFrame({
        "office": z["work"] - z["offday_ratio"] - 0.5 * z["night"],
        "residential": z["evening"] + z["night"] - z["work"] + 0.3 * z["offday_ratio"],
        "leisure": 1.5 * z["offday_ratio"] + 0.5 * z["evening"],
        "lowactivity": -z["level"] - 0.7 * z["amplitude"],
    })
    out = []
    for c, row in df.iterrows():
        s = scores.loc[c]
        best = s.idxmax()
        if len(df) < 2 or s.max() < 0.6:
            best = "mixed"
        name, why = ZONES[best]
        out.append({**{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()},
                    "cluster": int(c), "zone": name, "zone_key": best, "reason": why,
                    "scores": {k: round(float(v), 3) for k, v in s.items()}})
    return out


def run(ds: Dataset, method: str, feature: str, norm: str, params: dict, embed: str = "pca") -> dict:
    X = build_features(ds, feature, norm)
    labels = np.asarray(_fit(method, X, params))
    # relabel so cluster 0 is the largest (stable colors across runs)
    uniq = [c for c in pd.Series(labels[labels >= 0]).value_counts().index]
    remap = {c: i for i, c in enumerate(uniq)}
    labels = np.array([remap.get(c, -1) for c in labels])

    if embed == "tsne" and len(X) > 5:
        perplexity = float(min(30, max(2, (len(X) - 1) // 3)))
        E = TSNE(2, perplexity=perplexity, random_state=0, init="pca").fit_transform(X)
        ev = None
    else:
        pca = PCA(2, random_state=0).fit(X)
        E = pca.transform(X)
        ev = _r(pca.explained_variance_ratio_, 4)

    _, wd, we, wk = _profiles(ds)
    shape = lambda M: M / np.maximum(M.mean(axis=1, keepdims=True), 1e-9)
    wds, wes, wks = shape(wd.values), shape(we.values), shape(wk.values)
    centers = {}
    for c in sorted(set(labels)):
        m = labels == c
        centers[int(c)] = {
            "workday": _r(wds[m].mean(0), 4),
            "offday": _r(wes[m].mean(0), 4),
            "weekly": _r(wks[m].mean(0), 4),
            "workday_std": _r(wds[m].std(0), 4),
        }
    return {
        "labels": labels.tolist(),
        "regions": [{"id": r, "label": ds.labels[r]} for r in ds.regions],
        "embedding": _r(E, 4),
        "explained_variance": ev,
        "scores": _scores(X, labels),
        "centers": centers,
        "zones": interpret(ds, labels),
        "n_noise": int((labels < 0).sum()),
    }


def elbow(ds: Dataset, feature: str, norm: str, kmax: int = 10) -> dict:
    X = build_features(ds, feature, norm)
    ks, inertia, sil, ch, db = [], [], [], [], []
    for k in range(2, min(kmax, len(X) - 1) + 1):
        km = skc.KMeans(k, n_init=10, random_state=0).fit(X)
        ks.append(k)
        inertia.append(float(km.inertia_))
        sil.append(float(skm.silhouette_score(X, km.labels_)))
        ch.append(float(skm.calinski_harabasz_score(X, km.labels_)))
        db.append(float(skm.davies_bouldin_score(X, km.labels_)))
    return {"k": ks, "inertia": inertia, "silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db}
