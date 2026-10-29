"""Sliding-window forecasting dataset construction."""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .data import load
from .eda import _r, is_offday


@dataclass
class WindowConfig:
    dataset: str = "milano"
    regions: list = field(default_factory=list)  # empty = all
    freq: str = "1h"
    input_len: int = 24
    output_len: int = 12
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    train_stride: int = 1
    eval_stride: int = 1
    scaler: str = "zscore"  # zscore | minmax | none
    log1p: bool = False
    name: str = ""


class WindowData:
    """Holds the scaled multivariate series plus window start indices for each split.

    A window starting at s uses inputs [s, s+L) and targets [s+L, s+L+H).
    Validation/test windows may reach back into the previous segment for their
    inputs (that history is observed), but their targets lie fully inside the split.
    """

    def __init__(self, cfg: WindowConfig):
        ds = load(cfg.dataset)
        df = ds.resampled(cfg.freq)
        regions = [r for r in cfg.regions if r in ds.regions] or list(ds.regions)
        df = df[regions]
        self.id = uuid.uuid4().hex[:8]
        self.created = time.time()
        self.cfg = cfg
        self.ds = ds
        self.regions = regions
        self.labels = [ds.labels[r] for r in regions]
        self.times = df.index
        self.raw = df.values.astype(np.float64)
        self.T, self.N = self.raw.shape
        self.L, self.H = int(cfg.input_len), int(cfg.output_len)
        self.steps_per_day = ds.steps_per_day(cfg.freq)
        ridx = [ds.regions.index(r) for r in regions]
        self.coords = ds.centroids[ridx]

        T = self.T
        self.train_end = int(round(T * cfg.train_ratio))
        self.val_end = int(round(T * (cfg.train_ratio + cfg.val_ratio)))
        L, H = self.L, self.H
        if self.train_end < L + H + 1:
            raise ValueError("训练集太短：请减小输入/输出长度或增大训练比例")
        self.train_starts = np.arange(0, self.train_end - L - H + 1, max(1, cfg.train_stride))
        self.val_starts = np.arange(max(0, self.train_end - L), self.val_end - L - H + 1, max(1, cfg.eval_stride))
        self.test_starts = np.arange(max(0, self.val_end - L), T - L - H + 1, max(1, cfg.eval_stride))
        if len(self.test_starts) == 0:
            raise ValueError("测试集没有完整窗口：请减小输入/输出长度或减小训练+验证比例")

        # --- scaling (fit on training segment only) ---
        x = np.log1p(self.raw) if cfg.log1p else self.raw.copy()
        tr = x[: self.train_end]
        if cfg.scaler == "zscore":
            self.a, self.b = tr.mean(0), np.maximum(tr.std(0), 1e-8)
        elif cfg.scaler == "minmax":
            self.a, self.b = tr.min(0), np.maximum(tr.max(0) - tr.min(0), 1e-8)
        else:
            self.a, self.b = np.zeros(self.N), np.ones(self.N)
        self.scaled = ((x - self.a) / self.b).astype(np.float32)

        # --- calendar features: sin/cos time-of-day, sin/cos day-of-week, off-day flag ---
        idx = self.times
        tod = (idx.hour * 60 + idx.minute).values / 1440.0
        dow = idx.dayofweek.values / 7.0
        self.tf = np.c_[np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod), np.sin(2 * np.pi * dow),
                        np.cos(2 * np.pi * dow), is_offday(ds, idx).astype(float)].astype(np.float32)

    # scaled <-> original, node axis last
    def inverse(self, z: np.ndarray) -> np.ndarray:
        x = z * self.b + self.a
        return np.expm1(x) if self.cfg.log1p else x

    def X(self, starts) -> np.ndarray:
        return self.scaled[np.asarray(starts)[:, None] + np.arange(self.L)]  # (n, L, N)

    def Y(self, starts, scaled=True) -> np.ndarray:
        idx = np.asarray(starts)[:, None] + self.L + np.arange(self.H)
        return self.scaled[idx] if scaled else self.raw[idx]

    def split_starts(self, split: str) -> np.ndarray:
        return {"train": self.train_starts, "val": self.val_starts, "test": self.test_starts}[split]

    def summary(self) -> dict:
        tot = self.raw.sum(1)
        step = max(1, self.T // 1500)
        t = lambda i: str(self.times[min(i, self.T - 1)])
        ex = int(self.test_starts[len(self.test_starts) // 2])
        return {
            "id": self.id,
            "config": asdict(self.cfg),
            "dataset_title": self.ds.title,
            "n_regions": self.N,
            "n_steps": self.T,
            "steps_per_day": self.steps_per_day,
            "regions": [{"id": r, "label": l} for r, l in zip(self.regions, self.labels)],
            "splits": {
                "train": {"windows": len(self.train_starts), "from": t(0), "to": t(self.train_end - 1),
                          "steps": self.train_end},
                "val": {"windows": len(self.val_starts), "from": t(self.train_end), "to": t(self.val_end - 1),
                        "steps": self.val_end - self.train_end},
                "test": {"windows": len(self.test_starts), "from": t(self.val_end), "to": t(self.T - 1),
                         "steps": self.T - self.val_end},
            },
            "shapes": {
                "X_train": [len(self.train_starts), self.L, self.N],
                "Y_train": [len(self.train_starts), self.H, self.N],
                "X_test": [len(self.test_starts), self.L, self.N],
                "Y_test": [len(self.test_starts), self.H, self.N],
                "pooled_samples_train": len(self.train_starts) * self.N,
            },
            "preview": {
                "t": [str(x) for x in self.times[::step]],
                "v": _r(tot[::step], 1),
                "train_end": t(self.train_end), "val_end": t(self.val_end),
            },
            "example": {
                "start": ex,
                "t_in": [str(x) for x in self.times[ex: ex + self.L]],
                "t_out": [str(x) for x in self.times[ex + self.L: ex + self.L + self.H]],
                "v_in": _r(tot[ex: ex + self.L], 1),
                "v_out": _r(tot[ex + self.L: ex + self.L + self.H], 1),
            },
        }


DATASTORE: dict = {}
STORE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "windows.json")


def _persist():
    os.makedirs(os.path.dirname(STORE_FILE), exist_ok=True)
    items = [{"id": w.id, "created": w.created, "config": asdict(w.cfg)} for w in DATASTORE.values()]
    with open(STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)


def build(cfg: dict, wid: str | None = None, created: float | None = None, persist: bool = True) -> WindowData:
    fields = WindowConfig.__dataclass_fields__
    c = WindowConfig(**{k: v for k, v in cfg.items() if k in fields})
    wd = WindowData(c)
    if wid:
        wd.id = wid
    if created:
        wd.created = created
    if not c.name:
        c.name = f"{c.dataset} {c.freq} L{c.input_len}→H{c.output_len} {wd.N}区"
    DATASTORE[wd.id] = wd
    if persist:
        _persist()
    return wd


def remove(wid: str):
    if DATASTORE.pop(wid, None) is not None:
        _persist()


def load_saved():
    """Rebuild forecasting datasets defined in earlier sessions (cheap: just windows + scaling)."""
    if not os.path.exists(STORE_FILE):
        return
    try:
        with open(STORE_FILE, encoding="utf-8") as f:
            items = json.load(f)
    except Exception:
        return
    for it in items:
        try:
            build(it["config"], wid=it["id"], created=it.get("created"), persist=False)
        except Exception:
            pass
