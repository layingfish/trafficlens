"""Model interface, parameter schema helpers and the training monitor."""
from __future__ import annotations

import time

import numpy as np

REGISTRY: dict = {}


class StopTraining(Exception):
    """Raised inside fit() when the user presses stop."""


def P(name, label, type_, default, **kw):
    """Parameter spec rendered as a form control by the frontend."""
    return {"name": name, "label": label, "type": type_, "default": default, **kw}


def register(cls):
    REGISTRY[cls.key] = cls
    return cls


class Monitor:
    """Collects training events; the API streams them to the browser via SSE."""

    def __init__(self, job):
        self.job = job
        self._last_progress = 0.0

    def _emit(self, ev: dict):
        ev["ts"] = round(time.time() - self.job.started, 3)
        self.job.events.append(ev)

    def log(self, msg: str):
        self._emit({"type": "log", "msg": msg})

    def progress(self, frac: float, stage: str = "", force: bool = False):
        now = time.time()
        if force or now - self._last_progress > 0.25 or frac >= 1:
            self._last_progress = now
            self.job.progress = float(min(max(frac, 0), 1))
            self._emit({"type": "progress", "value": self.job.progress, "stage": stage})

    def curve(self, step: int, **metrics):
        """One point of the live training curve (epoch / iteration / tree count...)."""
        clean = {k: (None if v is None or not np.isfinite(v) else float(v)) for k, v in metrics.items()}
        self.job.curve.append({"step": step, **clean})
        self._emit({"type": "curve", "step": step, **clean})

    def check(self):
        if self.job.stop_flag:
            raise StopTraining()

    @property
    def stopping(self) -> bool:
        return self.job.stop_flag


class BaseModel:
    key = ""
    name = ""
    category = ""  # 统计 / 机器学习 / 深度学习 / 图神经网络
    desc = ""
    params: list = []

    def __init__(self, params: dict | None = None):
        self.p = {spec["name"]: spec["default"] for spec in self.all_params()}
        for k, v in (params or {}).items():
            if k in self.p:
                self.p[k] = v
        for spec in self.all_params():
            n = spec["name"]
            if spec["type"] == "int":
                self.p[n] = int(self.p[n])
            elif spec["type"] == "float":
                self.p[n] = float(self.p[n])
            elif spec["type"] == "bool":
                self.p[n] = self.p[n] in (True, "true", "1", 1, "on")

    @classmethod
    def all_params(cls) -> list:
        return list(cls.params)

    @classmethod
    def spec(cls) -> dict:
        return {"key": cls.key, "name": cls.name, "category": cls.category, "desc": cls.desc,
                "params": cls.all_params()}

    def fit(self, data, mon: Monitor):
        raise NotImplementedError

    def predict(self, data, starts) -> np.ndarray:
        """Return (n, H, N) predictions in scaled space."""
        raise NotImplementedError


def val_loss(model, data, max_windows: int = 400):
    """MSE (scaled) + MAE (original units) on the validation windows."""
    vs = data.val_starts
    if len(vs) == 0:
        return None, None
    if len(vs) > max_windows:
        vs = vs[np.linspace(0, len(vs) - 1, max_windows).astype(int)]
    pz = model.predict(data, vs)
    yz = data.Y(vs)
    mse = float(((pz - yz) ** 2).mean())
    mae = float(np.abs(data.inverse(pz) - data.Y(vs, scaled=False)).mean())
    return mse, mae
