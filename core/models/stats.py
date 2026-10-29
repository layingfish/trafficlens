"""Classical statistical forecasters, vectorised over windows and regions."""
from __future__ import annotations

import warnings

import numpy as np

from .base import BaseModel, P, register, val_loss


def _period(data, which: str) -> int:
    return {"day": data.steps_per_day, "week": data.steps_per_day * 7, "last": 1}[which]


@register
class HistoricalAverage(BaseModel):
    key = "ha"
    name = "历史平均 HA"
    category = "统计模型"
    desc = "用过去 K 个周期中同一相位（同一时刻）的观测均值作为预测，经典交通预测基线。"
    params = [
        P("period", "周期", "select", "week", options=["day", "week"]),
        P("n_periods", "回看周期数 K", "int", 4, min=1, max=20),
    ]

    def fit(self, data, mon):
        mon.log(f"HA 无需训练：周期={self.p['period']}，回看 {self.p['n_periods']} 个周期")
        mse, mae = val_loss(self, data)
        mon.curve(1, val_loss=mse, val_mae=mae)
        mon.progress(1.0, "完成")

    def predict(self, data, starts):
        m, K = _period(data, self.p["period"]), self.p["n_periods"]
        starts = np.asarray(starts)
        origin = starts + data.L - 1
        h = np.arange(1, data.H + 1)
        k0 = np.ceil(h / m).astype(int)  # first lag (in periods) that is already observed
        out = np.zeros((len(starts), data.H, data.N), dtype=np.float32)
        cnt = np.zeros((len(starts), data.H, 1), dtype=np.float32)
        for j in range(K):
            idx = origin[:, None] + h[None, :] - (k0 + j)[None, :] * m
            ok = idx >= 0
            out += np.where(ok[..., None], data.scaled[np.clip(idx, 0, None)], 0)
            cnt += ok[..., None]
        last = data.scaled[origin][:, None, :]
        return np.where(cnt > 0, out / np.maximum(cnt, 1), last)


@register
class SeasonalNaive(BaseModel):
    key = "snaive"
    name = "季节朴素 Seasonal Naive"
    category = "统计模型"
    desc = "预测值 = 上一个季节周期同一时刻的观测；季节取 last 时退化为“最后值持续”。"
    params = [P("season", "季节长度", "select", "day", options=["last", "day", "week"])]

    def fit(self, data, mon):
        mon.log(f"季节朴素：季节={self.p['season']}")
        mse, mae = val_loss(self, data)
        mon.curve(1, val_loss=mse, val_mae=mae)
        mon.progress(1.0, "完成")

    def predict(self, data, starts):
        m = _period(data, self.p["season"])
        origin = np.asarray(starts) + data.L - 1
        h = np.arange(1, data.H + 1)
        idx = origin[:, None] + h[None, :] - np.ceil(h / m).astype(int)[None, :] * m
        return data.scaled[np.clip(idx, 0, None)]


@register
class ARIMAModel(BaseModel):
    key = "arima"
    name = "ARIMA"
    category = "统计模型"
    desc = ("每个区域单独拟合 ARIMA(p,d,q)；预测时把 ARMA 转为截断 AR(∞) 表示，"
            "对所有测试窗口做向量化的多步递推预测。")
    params = [
        P("p", "AR 阶数 p", "int", 3, min=0, max=12),
        P("d", "差分阶数 d", "int", 0, min=0, max=2),
        P("q", "MA 阶数 q", "int", 1, min=0, max=6),
        P("fit_len", "拟合使用的训练末段长度", "int", 400, min=50, max=5000),
        P("maxiter", "最大迭代次数", "int", 50, min=10, max=500),
    ]

    def fit(self, data, mon):
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.arima_process import arma2ar

        p, d, q = self.p["p"], self.p["d"], self.p["q"]
        if data.L <= d + 1:
            raise ValueError("输入长度必须大于差分阶数 d+1")
        K = data.L - d
        y_all = data.scaled[: data.train_end]
        y_all = y_all[-self.p["fit_len"]:]
        self.coef = np.zeros((data.N, K - 1))
        self.mu = np.zeros(data.N)
        fails = 0
        for n in range(data.N):
            mon.check()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = ARIMA(y_all[:, n].astype(float), order=(p, d, q), trend="c" if d == 0 else "n").fit(
                        method_kwargs={"maxiter": self.p["maxiter"]})
                pr = dict(zip(res.param_names, res.params))
                phi = np.array([pr.get(f"ar.L{i}", 0.0) for i in range(1, p + 1)])
                theta = np.array([pr.get(f"ma.L{i}", 0.0) for i in range(1, q + 1)])
                c = arma2ar(np.r_[1, -phi], np.r_[1, theta], lags=K)
                self.coef[n] = -c[1:K]
                self.mu[n] = pr.get("const", 0.0)
            except Exception as e:  # fall back to persistence for this region
                fails += 1
                self.coef[n] = 0
                if d == 0:
                    self.coef[n, 0] = 1
                mon.log(f"区域 {data.labels[n]} 拟合失败（{type(e).__name__}），退化为持续预测")
            mon.progress((n + 1) / data.N, f"拟合区域 {n + 1}/{data.N}")
            if (n + 1) % max(1, data.N // 10) == 0:
                mon.log(f"已拟合 {n + 1}/{data.N} 个区域")
        mse, mae = val_loss(self, data)
        mon.curve(1, val_loss=mse, val_mae=mae)
        mon.log(f"ARIMA({p},{d},{q}) 拟合完成，失败 {fails} 个")

    def predict(self, data, starts):
        d, H = self.p["d"], data.H
        X = data.X(starts).astype(np.float64)  # (n, L, N)
        levels = [X]
        for _ in range(d):
            levels.append(np.diff(levels[-1], axis=1))
        z = levels[-1] - self.mu  # (n, K, N)
        K1 = self.coef.shape[1]
        buf = z[:, -K1:, :] if K1 > 0 else z[:, :0, :]
        rev = self.coef[:, ::-1].T  # (K1, N): weight for oldest..newest
        fc = []
        for _ in range(H):
            nxt = (buf * rev[None]).sum(1) if K1 > 0 else np.zeros(z.shape[::2])
            fc.append(nxt)
            buf = np.concatenate([buf[:, 1:], nxt[:, None]], axis=1) if K1 > 0 else buf
        f = np.stack(fc, 1) + self.mu  # (n, H, N) at deepest differencing level
        for lv in range(d - 1, -1, -1):
            f = levels[lv][:, -1:, :] + np.cumsum(f, axis=1)
        return f.astype(np.float32)


@register
class HoltWinters(BaseModel):
    key = "holtwinters"
    name = "Holt-Winters 指数平滑"
    category = "统计模型"
    desc = ("加法三次指数平滑（水平+趋势+季节）。状态沿整条序列滚动更新，"
            "在每个预测起点以当前状态外推。可手动设定平滑系数或按区域自动优化。")
    params = [
        P("season", "季节周期", "select", "day", options=["day", "week"]),
        P("alpha", "水平平滑 α", "float", 0.3, min=0.01, max=0.99, step=0.01),
        P("beta", "趋势平滑 β", "float", 0.01, min=0.0, max=0.99, step=0.01),
        P("gamma", "季节平滑 γ", "float", 0.2, min=0.0, max=0.99, step=0.01),
        P("trend", "包含趋势项", "bool", False),
        P("optimize", "自动优化 α/β/γ（逐区域）", "bool", False),
    ]

    def fit(self, data, mon):
        m = _period(data, self.p["season"])
        if data.train_end < 2 * m:
            raise ValueError("训练段不足两个季节周期，请改用 day 季节或增大训练比例")
        N = data.N
        al = np.full(N, self.p["alpha"])
        be = np.full(N, self.p["beta"] if self.p["trend"] else 0.0)
        ga = np.full(N, self.p["gamma"])
        if self.p["optimize"]:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            tr = data.scaled[: data.train_end].astype(float)
            for n in range(N):
                mon.check()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        r = ExponentialSmoothing(tr[:, n], trend="add" if self.p["trend"] else None,
                                                 seasonal="add", seasonal_periods=m).fit()
                    al[n] = r.params["smoothing_level"]
                    ga[n] = r.params["smoothing_seasonal"] or 0.0
                    if self.p["trend"]:
                        be[n] = r.params["smoothing_trend"] or 0.0
                except Exception:
                    pass
                mon.progress((n + 1) / N * 0.9, f"优化区域 {n + 1}/{N}")
            mon.log(f"自动优化完成：α 中位数 {np.median(al):.3f}，γ 中位数 {np.median(ga):.3f}")
        self.m = m
        self._filter(data, al, be, ga)
        mse, mae = val_loss(self, data)
        mon.curve(1, val_loss=mse, val_mae=mae)
        mon.progress(1.0, "完成")

    def _filter(self, data, al, be, ga):
        y = data.scaled.astype(np.float64)
        T, N, m = y.shape[0], y.shape[1], self.m
        Lv, Bv, Sv = np.zeros((T, N)), np.zeros((T, N)), np.zeros((T, N))
        l0 = y[:m].mean(0)
        b0 = (y[m:2 * m].mean(0) - l0) / m if self.p["trend"] else np.zeros(N)
        Sv[:m] = y[:m] - l0
        Lv[m - 1], Bv[m - 1] = l0, b0
        for t in range(m, T):
            lp, bp = Lv[t - 1], Bv[t - 1]
            Lv[t] = al * (y[t] - Sv[t - m]) + (1 - al) * (lp + bp)
            Bv[t] = be * (Lv[t] - lp) + (1 - be) * bp
            Sv[t] = ga * (y[t] - Lv[t]) + (1 - ga) * Sv[t - m]
        self.Lv, self.Bv, self.Sv = Lv, Bv, Sv

    def predict(self, data, starts):
        o = np.asarray(starts) + data.L - 1
        h = np.arange(1, data.H + 1)
        sidx = o[:, None] + h[None, :] - np.ceil(h / self.m).astype(int)[None, :] * self.m
        oc = np.maximum(o, self.m - 1)
        f = self.Lv[oc][:, None, :] + h[None, :, None] * self.Bv[oc][:, None, :] + self.Sv[np.clip(sidx, 0, None)]
        naive = data.scaled[np.clip(sidx, 0, None)]
        return np.where((o >= self.m - 1)[:, None, None], f, naive).astype(np.float32)
