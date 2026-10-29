"""Dataset registry, loading, resampling and geometry helpers."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

DATASETS = {
    "milano": {
        "title": "米兰 Milano（Telecom Italia，88 个 NIL 街区）",
        "csv": "milano_traffic_nid.csv",
        "geo": "milano.geojson",
        "key": "NIL",
        "label": "NIL",
        "freq": "10min",
        "unit": "流量活动量",
    },
    "trentino": {
        "title": "特伦蒂诺 Trentino（Telecom Italia，223 个市镇）",
        "csv": "trentino_traffic_nid.csv",
        "geo": "trento.geojson",
        "key": "NAME_3",
        "label": "NAME_3",
        "freq": "10min",
        "unit": "流量活动量",
    },
    "zte": {
        "title": "济南 Jinan（ZTE 4G 下行，148 个街道）",
        "csv": "zte_4g_down.region.csv",
        "geo": "zte_4g_down.traffic.geojson",
        "key": "unique_id",
        "label": "name",
        "freq": "15min",
        "unit": "下行流量",
    },
}

FREQ_OPTIONS = ["native", "30min", "1h", "2h", "4h", "1D"]


@dataclass
class Dataset:
    key: str
    title: str
    unit: str
    freq: str
    df: pd.DataFrame  # (T, N) native resolution
    regions: list
    labels: dict
    centroids: np.ndarray  # (N, 2) lon, lat
    geo: dict
    _cache: dict = field(default_factory=dict)

    def resampled(self, freq: str = "native") -> pd.DataFrame:
        if freq in (None, "", "native", self.freq):
            return self.df
        if freq not in self._cache:
            self._cache[freq] = self.df.resample(freq).sum()
        return self._cache[freq]

    def steps_per_day(self, freq: str = "native") -> int:
        f = self.freq if freq in (None, "", "native") else freq
        return int(round(pd.Timedelta("1D") / pd.Timedelta(f)))

    def meta(self) -> dict:
        return {
            "key": self.key,
            "title": self.title,
            "unit": self.unit,
            "freq": self.freq,
            "n_regions": len(self.regions),
            "n_steps": len(self.df),
            "start": str(self.df.index[0]),
            "end": str(self.df.index[-1]),
            "regions": [{"id": r, "label": self.labels[r]} for r in self.regions],
            "freq_options": [f for f in FREQ_OPTIONS if f == "native" or pd.Timedelta(f) >= pd.Timedelta(self.freq)],
        }


def _ring_area_centroid(ring):
    pts = np.asarray(ring, dtype=float)[:, :2]
    x, y = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    a = cross.sum() / 2.0
    if abs(a) < 1e-15:
        return 0.0, (x.mean(), y.mean())
    cx = ((x + x1) * cross).sum() / (6 * a)
    cy = ((y + y1) * cross).sum() / (6 * a)
    return abs(a), (cx, cy)


def geometry_centroid(geom) -> tuple:
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    best = max((_ring_area_centroid(p[0]) for p in polys), key=lambda t: t[0])
    return best[1]


@lru_cache(maxsize=None)
def load(key: str) -> Dataset:
    cfg = DATASETS[key]
    df = pd.read_csv(os.path.join(DATA_DIR, cfg["csv"]), index_col=0)
    df.index = pd.to_datetime(df.index)
    df.columns = [str(c) for c in df.columns]
    df = df.astype("float64").clip(lower=0)
    with open(os.path.join(DATA_DIR, cfg["geo"]), encoding="utf-8") as f:
        geo = json.load(f)
    by_id = {str(ft["properties"][cfg["key"]]): ft for ft in geo["features"]}
    regions = [c for c in df.columns if c in by_id]
    df = df[regions]
    labels, cents, feats = {}, [], []
    means = df.mean()
    for r in regions:
        ft = by_id[r]
        labels[r] = str(ft["properties"].get(cfg["label"], r))
        cents.append(geometry_centroid(ft["geometry"]))
        feats.append({
            "type": "Feature",
            "geometry": ft["geometry"],
            "properties": {"rid": r, "label": labels[r], "mean": float(means[r])},
        })
    # disambiguate duplicate labels (e.g. same street name in two districts)
    seen = {}
    for r in regions:
        seen.setdefault(labels[r], []).append(r)
    for lab, rs in seen.items():
        if len(rs) > 1:
            for r in rs:
                labels[r] = f"{lab}({r})"
    for ft in feats:
        ft["properties"]["label"] = labels[ft["properties"]["rid"]]
    return Dataset(
        key=key,
        title=cfg["title"],
        unit=cfg["unit"],
        freq=cfg["freq"],
        df=df,
        regions=regions,
        labels=labels,
        centroids=np.asarray(cents),
        geo={"type": "FeatureCollection", "features": feats},
    )


def haversine_matrix(lonlat: np.ndarray) -> np.ndarray:
    lon, lat = np.radians(lonlat[:, 0]), np.radians(lonlat[:, 1])
    dlon = lon[:, None] - lon[None, :]
    dlat = lat[:, None] - lat[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def knn_adjacency(lonlat: np.ndarray, k: int = 6) -> np.ndarray:
    """Symmetric binary kNN adjacency (no self loops)."""
    d = haversine_matrix(lonlat)
    n = len(d)
    k = max(1, min(k, n - 1))
    A = np.zeros((n, n))
    idx = np.argsort(d, axis=1)[:, 1:k + 1]
    A[np.repeat(np.arange(n), k), idx.ravel()] = 1
    return np.maximum(A, A.T)


def morans_i(values: np.ndarray, W: np.ndarray) -> float:
    z = values - values.mean()
    s0 = W.sum()
    denom = (z ** 2).sum()
    if s0 == 0 or denom == 0:
        return 0.0
    return float(len(values) / s0 * (z @ W @ z) / denom)
