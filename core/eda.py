"""Exploratory analysis: temporal & spatial dynamics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Dataset, haversine_matrix, knn_adjacency, morans_i

# Public holidays inside each dataset's time range (treated as non-working days).
HOLIDAYS = {
    "milano": ["2013-11-01", "2013-12-07", "2013-12-08", "2013-12-25", "2013-12-26", "2014-01-01"],
    "trentino": ["2013-11-01", "2013-12-08", "2013-12-25", "2013-12-26", "2014-01-01"],
    "zte": [],
}


def _r(a, nd=3):
    """Round arrays/scalars for compact JSON."""
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def is_offday(ds: Dataset, index: pd.DatetimeIndex) -> np.ndarray:
    hol = set(pd.to_datetime(HOLIDAYS.get(ds.key, [])).date)
    return np.array([(t.weekday() >= 5) or (t.date() in hol) for t in index])


def hourly(ds: Dataset) -> pd.DataFrame:
    return ds.resampled("1h")


def region_stats(ds: Dataset) -> pd.DataFrame:
    h = hourly(ds)
    off = is_offday(ds, h.index)
    hod = h.index.hour
    wd = h[~off].groupby(hod[~off]).mean()
    we = h[off].groupby(hod[off]).mean()
    mean = h.mean()
    out = pd.DataFrame({
        "mean": mean,
        "std": h.std(),
        "max": h.max(),
        "min": h.min(),
    })
    out["cv"] = out["std"] / out["mean"].replace(0, np.nan)
    out["peak_hour"] = wd.idxmax()
    out["offday_ratio"] = we.mean() / wd.mean().replace(0, np.nan)
    out["night_share"] = wd.loc[0:5].mean() / wd.mean().replace(0, np.nan)
    out["day_night_ratio"] = wd.loc[9:18].mean() / wd.loc[0:5].mean().replace(0, np.nan)
    return out.fillna(0)


def overview(ds: Dataset) -> dict:
    h = hourly(ds)
    total = h.sum(axis=1)
    daily = ds.df.resample("1D").sum().sum(axis=1)
    st = region_stats(ds)
    top = st.sort_values("mean", ascending=False)
    return {
        "meta": ds.meta(),
        "total_hourly": {"t": [str(t) for t in total.index], "v": _r(total, 1)},
        "total_daily": {"t": [str(t.date()) for t in daily.index], "v": _r(daily, 1),
                        "offday": is_offday(ds, daily.index).tolist()},
        "kpi": {
            "total": float(ds.df.values.sum()),
            "mean_per_region_hour": float(h.values.mean()),
            "peak_time": str(total.idxmax()),
            "peak_value": float(total.max()),
            "trough_time": str(total.idxmin()),
            "cv_median": float(st["cv"].median()),
        },
        "table": [
            {"id": r, "label": ds.labels[r], **{k: float(v) for k, v in row.items()}}
            for r, row in top.iterrows()
        ],
    }


def temporal(ds: Dataset, regions: list, freq: str = "1h") -> dict:
    regions = [r for r in regions if r in ds.regions] or list(ds.regions[:3])
    df = ds.resampled(freq)[regions]
    h = hourly(ds)[regions]
    off = is_offday(ds, h.index)
    hod = h.index.hour

    profiles = {}
    for r in regions:
        s = h[r]
        profiles[r] = {
            "workday": _r(s[~off].groupby(hod[~off]).mean(), 2),
            "offday": _r(s[off].groupby(hod[off]).mean(), 2),
        }

    agg = h.sum(axis=1)
    heat = agg.groupby([agg.index.dayofweek, agg.index.hour]).mean().unstack().reindex(range(7)).fillna(0)

    # Hour-of-day distribution (box stats) for the aggregate of selected regions
    box = []
    for hr in range(24):
        v = agg[agg.index.hour == hr].values
        q = np.percentile(v, [0, 25, 50, 75, 100])
        box.append(_r(q, 1))

    # Multi-seasonal decomposition of the first region
    first = h[regions[0]]
    decomp = None
    try:
        from statsmodels.tsa.seasonal import MSTL
        periods = (24, 168) if len(first) >= 2 * 168 * 2 else (24,)
        res = MSTL(first.values, periods=periods).fit()
        seas = np.atleast_2d(res.seasonal.T) if res.seasonal.ndim > 1 else np.atleast_2d(res.seasonal)
        decomp = {
            "t": [str(t) for t in first.index],
            "observed": _r(first.values, 1),
            "trend": _r(res.trend, 1),
            "seasonal": {f"周期 {p}h": _r(seas[i], 1) for i, p in enumerate(periods)},
            "resid": _r(res.resid, 1),
        }
    except Exception as e:  # pragma: no cover - decomposition is optional
        decomp = {"error": str(e)}

    # Autocorrelation (hourly) up to two weeks
    from statsmodels.tsa.stattools import acf
    nlags = min(24 * 14, len(first) - 2)
    ac = {r: _r(acf(h[r].values, nlags=nlags, fft=True), 4) for r in regions}

    daily = ds.df[regions].resample("1D").sum()
    return {
        "regions": [{"id": r, "label": ds.labels[r]} for r in regions],
        "series": {"t": [str(t) for t in df.index], "v": {r: _r(df[r].values, 1) for r in regions}},
        "profiles": profiles,
        "heatmap": {"dow": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"], "values": _r(heat.values, 1)},
        "box": box,
        "decomp": decomp,
        "acf": ac,
        "daily": {"t": [str(t.date()) for t in daily.index], "v": {r: _r(daily[r].values, 1) for r in regions},
                  "offday": is_offday(ds, daily.index).tolist()},
    }


def local_moran(values: np.ndarray, W: np.ndarray, perms: int = 199, seed: int = 0) -> dict:
    """Local Moran's I (LISA) with conditional permutation significance."""
    n = len(values)
    Wr = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
    z = (values - values.mean()) / (values.std() + 1e-12)
    lag = Wr @ z
    Ii = z * lag
    rng = np.random.default_rng(seed)
    k = (W > 0).sum(axis=1)
    larger = np.zeros(n)
    for i in range(n):
        others = np.delete(z, i)
        ki = max(int(k[i]), 1)
        samp = np.array([rng.choice(others, ki, replace=False).mean() for _ in range(perms)])
        sim = z[i] * samp
        larger[i] = (np.abs(sim) >= abs(Ii[i])).sum()
    p = (larger + 1) / (perms + 1)
    quad = np.where(z > 0, np.where(lag > 0, "HH", "HL"), np.where(lag > 0, "LH", "LL"))
    cls = np.where(p < 0.05, quad, "ns")
    return {"I": _r(Ii, 4), "p": _r(p, 4), "cls": cls.tolist(), "z": _r(z, 3), "lag": _r(lag, 3)}


def spatial(ds: Dataset, knn: int = 6) -> dict:
    st = region_stats(ds)
    h = hourly(ds)
    off = is_offday(ds, h.index)
    hod = h.index.hour
    wd = h[~off].groupby(hod[~off]).mean()
    we = h[off].groupby(hod[off]).mean()
    W = knn_adjacency(ds.centroids, knn)
    mean_vals = st["mean"].values
    moran_hour = [morans_i(np.log1p(wd.loc[hr].values), W) for hr in range(24)]
    daily = ds.df.resample("1D").sum()
    moran_day = [morans_i(np.log1p(daily.iloc[i].values), W) for i in range(len(daily))]
    lisa = local_moran(np.log1p(mean_vals), W)

    # Correlation vs distance (distance-decay of temporal co-movement)
    corr = np.corrcoef(h.values.T)
    dist = haversine_matrix(ds.centroids)
    iu = np.triu_indices(len(ds.regions), 1)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(iu[0]), min(3000, len(iu[0])), replace=False)
    pairs = np.c_[dist[iu][sel], corr[iu][sel]]

    return {
        "regions": [{"id": r, "label": ds.labels[r]} for r in ds.regions],
        "metrics": {k: _r(st[k].values, 4) for k in st.columns},
        "hour_profile": {"workday": _r(wd.values, 1), "offday": _r(we.values, 1)},  # (24, N)
        "daily": {"t": [str(t.date()) for t in daily.index], "values": _r(daily.values, 1)},  # (D, N)
        "moran": {
            "global_mean": morans_i(np.log1p(mean_vals), W),
            "by_hour": _r(moran_hour, 4),
            "by_day": _r(moran_day, 4),
            "days": [str(t.date()) for t in daily.index],
        },
        "lisa": lisa,
        "distance_decay": _r(pairs, 3),
    }


def snapshot(ds: Dataset, freq: str, i: int) -> dict:
    df = ds.resampled(freq)
    i = int(np.clip(i, 0, len(df) - 1))
    return {"t": str(df.index[i]), "i": i, "n": len(df), "values": _r(df.iloc[i].values, 1)}
