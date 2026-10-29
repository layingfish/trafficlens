"""Classical machine-learning forecasters.

All regions share one model: every (window, region) pair is a sample whose features
are the L scaled lags plus calendar features of the forecast origin (direct
multi-output strategy, H targets).
"""
from __future__ import annotations

import time

import numpy as np

from .base import BaseModel, P, register, val_loss

COMMON = [
    P("use_time_features", "加入时间特征（时刻/星期/节假日）", "bool", True),
    P("max_train_samples", "最大训练样本数（窗口×区域）", "int", 60000, min=1000, max=2000000, step=1000),
    P("seed", "随机种子", "int", 42, min=0, max=99999),
]


def pairs(starts, N):
    starts = np.asarray(starts)
    return np.repeat(starts, N), np.tile(np.arange(N), len(starts))


def features(data, t, n, use_tf=True):
    X = data.scaled[t[:, None] + np.arange(data.L), n[:, None]]
    if use_tf:
        X = np.c_[X, data.tf[t + data.L]]  # calendar of first target step
    return X


def targets(data, t, n):
    return data.scaled[t[:, None] + data.L + np.arange(data.H), n[:, None]]


class PooledModel(BaseModel):
    category = "机器学习"

    @classmethod
    def all_params(cls):
        return list(cls.params) + COMMON

    def _train_set(self, data, mon):
        t, n = pairs(data.train_starts, data.N)
        rng = np.random.default_rng(self.p["seed"])
        if len(t) > self.p["max_train_samples"]:
            sel = rng.choice(len(t), self.p["max_train_samples"], replace=False)
            t, n = t[sel], n[sel]
        X, Y = features(data, t, n, self.p["use_time_features"]), targets(data, t, n)
        mon.log(f"训练样本 {X.shape[0]:,} 条，特征维度 {X.shape[1]}，输出维度 {Y.shape[1]}")
        return X, Y

    def _val_set(self, data, max_windows=300):
        vs = data.val_starts
        if len(vs) > max_windows:
            vs = vs[np.linspace(0, len(vs) - 1, max_windows).astype(int)]
        t, n = pairs(vs, data.N)
        return features(data, t, n, self.p["use_time_features"]), targets(data, t, n)

    def _predict_rows(self, X):
        return self.model.predict(X)

    def predict(self, data, starts):
        t, n = pairs(starts, data.N)
        out = []
        for i in range(0, len(t), 50000):
            out.append(self._predict_rows(features(data, t[i:i + 50000], n[i:i + 50000], self.p["use_time_features"])))
        Yp = np.concatenate(out).reshape(len(starts), data.N, data.H)
        return Yp.transpose(0, 2, 1).astype(np.float32)


@register
class RidgeModel(PooledModel):
    key = "ridge"
    name = "线性回归 Ridge"
    desc = "带 L2 正则的线性自回归模型（直接多步输出），速度快、可解释。"
    params = [
        P("alpha", "正则强度 α", "float", 1.0, min=0.0, max=1000, step=0.1),
        P("fit_intercept", "拟合截距", "bool", True),
    ]

    def fit(self, data, mon):
        from sklearn.linear_model import Ridge
        X, Y = self._train_set(data, mon)
        mon.progress(0.3, "求解正规方程")
        self.model = Ridge(alpha=self.p["alpha"], fit_intercept=self.p["fit_intercept"]).fit(X, Y)
        tr = float(((self.model.predict(X) - Y) ** 2).mean())
        mse, mae = val_loss(self, data)
        mon.curve(1, train_loss=tr, val_loss=mse, val_mae=mae)
        mon.progress(1.0, "完成")


@register
class KNNModel(PooledModel):
    key = "knn"
    name = "K 近邻 KNN"
    desc = "在历史窗口中寻找最相似的 K 个片段，用其后续走势的（加权）平均作为预测。"
    params = [
        P("n_neighbors", "近邻数 K", "int", 10, min=1, max=200),
        P("weights", "权重", "select", "distance", options=["uniform", "distance"]),
        P("metric", "距离度量", "select", "euclidean", options=["euclidean", "manhattan", "cosine"]),
    ]

    def fit(self, data, mon):
        from sklearn.neighbors import KNeighborsRegressor
        X, Y = self._train_set(data, mon)
        algo = "brute" if self.p["metric"] == "cosine" else "auto"
        self.model = KNeighborsRegressor(self.p["n_neighbors"], weights=self.p["weights"], metric=self.p["metric"],
                                         algorithm=algo, n_jobs=-1).fit(X, Y)
        mon.progress(0.5, "构建索引完成，评估验证集")
        mse, mae = val_loss(self, data)
        mon.curve(1, val_loss=mse, val_mae=mae)
        mon.progress(1.0, "完成")


@register
class SVRModel(PooledModel):
    key = "svr"
    name = "支持向量回归 SVR"
    desc = "核支持向量回归，每个预测步长训练一个 SVR（直接多步策略）。核方法复杂度高，建议限制样本数。"
    params = [
        P("kernel", "核函数", "select", "rbf", options=["rbf", "linear", "poly", "sigmoid"]),
        P("C", "惩罚系数 C", "float", 1.0, min=0.01, max=1000, step=0.01),
        P("epsilon", "不敏感带 ε", "float", 0.05, min=0.0, max=2, step=0.01),
        P("gamma", "核系数 γ", "select", "scale", options=["scale", "auto"]),
    ]

    @classmethod
    def all_params(cls):
        ps = super().all_params()
        return [dict(p, default=4000) if p["name"] == "max_train_samples" else p for p in ps]

    def fit(self, data, mon):
        from sklearn.svm import SVR
        X, Y = self._train_set(data, mon)
        Xv, Yv = self._val_set(data)
        self.models = []
        for h in range(data.H):
            mon.check()
            m = SVR(kernel=self.p["kernel"], C=self.p["C"], epsilon=self.p["epsilon"], gamma=self.p["gamma"])
            m.fit(X, Y[:, h])
            self.models.append(m)
            vl = float(((m.predict(Xv) - Yv[:, h]) ** 2).mean())
            tr = float(((m.predict(X[:2000]) - Y[:2000, h]) ** 2).mean())
            mon.curve(h + 1, train_loss=tr, val_loss=vl)
            mon.progress((h + 1) / data.H, f"步长 {h + 1}/{data.H}")
            mon.log(f"步长 h={h + 1}: 支持向量 {len(m.support_)}，验证 MSE={vl:.4f}")

    def _predict_rows(self, X):
        return np.stack([m.predict(X) for m in self.models], 1)


@register
class RandomForestModel(PooledModel):
    key = "rf"
    name = "随机森林 Random Forest"
    desc = "多棵决策树的 bagging 集成，原生支持多输出；按批增加树并实时报告验证误差。"
    params = [
        P("n_estimators", "树的数量", "int", 100, min=10, max=1000, step=10),
        P("max_depth", "最大深度（0=不限）", "int", 12, min=0, max=100),
        P("min_samples_leaf", "叶子最小样本数", "int", 5, min=1, max=200),
        P("max_features", "每次分裂特征比例", "float", 0.5, min=0.05, max=1.0, step=0.05),
        P("trees_per_step", "每步新增树数（监控粒度）", "int", 10, min=1, max=100),
    ]

    def fit(self, data, mon):
        from sklearn.ensemble import RandomForestRegressor
        X, Y = self._train_set(data, mon)
        Xv, Yv = self._val_set(data)
        self.model = RandomForestRegressor(
            n_estimators=0, warm_start=True, max_depth=self.p["max_depth"] or None,
            min_samples_leaf=self.p["min_samples_leaf"], max_features=self.p["max_features"],
            n_jobs=-1, random_state=self.p["seed"])
        total, step = self.p["n_estimators"], self.p["trees_per_step"]
        n = 0
        while n < total:
            if mon.stopping and n > 0:
                mon.log("收到停止指令，保留当前已训练的树")
                break
            n = min(total, n + step)
            self.model.set_params(n_estimators=n)
            self.model.fit(X, Y)
            vl = float(((self.model.predict(Xv) - Yv) ** 2).mean())
            mon.curve(n, val_loss=vl)
            mon.progress(n / total, f"{n}/{total} 棵树")
        mon.log(f"随机森林训练完成：{n} 棵树")


@register
class LightGBMModel(PooledModel):
    key = "lightgbm"
    name = "梯度提升树 LightGBM"
    desc = ("GBDT 直接多步预测：把预测步长 h 作为特征，单个模型覆盖全部步长。"
            "逐迭代回传训练/验证 L2 损失。")
    params = [
        P("n_estimators", "迭代轮数", "int", 300, min=10, max=5000, step=10),
        P("learning_rate", "学习率", "float", 0.05, min=0.001, max=1.0, step=0.001),
        P("num_leaves", "叶子数", "int", 63, min=4, max=1024),
        P("max_depth", "最大深度（-1=不限）", "int", -1, min=-1, max=64),
        P("subsample", "行采样比例", "float", 0.8, min=0.1, max=1.0, step=0.05),
        P("colsample_bytree", "列采样比例", "float", 0.8, min=0.1, max=1.0, step=0.05),
        P("reg_lambda", "L2 正则 λ", "float", 0.0, min=0.0, max=100, step=0.1),
        P("early_stopping", "早停轮数（0=关闭）", "int", 50, min=0, max=1000),
    ]

    @classmethod
    def all_params(cls):
        ps = super().all_params()
        return [dict(p, default=20000) if p["name"] == "max_train_samples" else p for p in ps]

    @staticmethod
    def _expand(X, Y, H):
        n = len(X)
        hcol = np.repeat(np.arange(H)[None], n, 0).reshape(-1, 1)
        Xe = np.c_[np.repeat(X, H, axis=0), hcol]
        return Xe, (Y.reshape(-1) if Y is not None else None)

    def fit(self, data, mon):
        import lightgbm as lgb
        X, Y = self._train_set(data, mon)
        Xv, Yv = self._val_set(data, 150)
        Xe, Ye = self._expand(X, Y, data.H)
        Xve, Yve = self._expand(Xv, Yv, data.H)
        mon.log(f"展开后训练行数 {len(Xe):,}（样本×步长）")
        total = self.p["n_estimators"]
        hist = {}

        def cb(env):
            it = env.iteration + 1
            if mon.stopping:
                raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)
            if it % max(1, total // 100) == 0 or it == total:
                vals = {r[0]: r[2] for r in env.evaluation_result_list}
                mon.curve(it, train_loss=vals.get("train"), val_loss=vals.get("val"))
                mon.progress(it / total, f"迭代 {it}/{total}")

        callbacks = [cb, lgb.record_evaluation(hist)]
        if self.p["early_stopping"] > 0:
            callbacks.append(lgb.early_stopping(self.p["early_stopping"], verbose=False))
        dtrain = lgb.Dataset(Xe, Ye)
        self.model = lgb.train(
            {"objective": "l2", "learning_rate": self.p["learning_rate"], "num_leaves": self.p["num_leaves"],
             "max_depth": self.p["max_depth"], "bagging_fraction": self.p["subsample"], "bagging_freq": 1,
             "feature_fraction": self.p["colsample_bytree"], "lambda_l2": self.p["reg_lambda"],
             "seed": self.p["seed"], "verbose": -1, "num_threads": 0},
            dtrain, num_boost_round=total,
            valid_sets=[dtrain, lgb.Dataset(Xve, Yve, reference=dtrain)], valid_names=["train", "val"],
            callbacks=callbacks)
        mon.log(f"LightGBM 完成：最佳迭代 {self.model.best_iteration or self.model.current_iteration()}")
        self.H = data.H

    def _predict_rows(self, X):
        Xe, _ = self._expand(X, None, self.H)
        it = self.model.best_iteration or None
        return self.model.predict(Xe, num_iteration=it).reshape(len(X), self.H)
