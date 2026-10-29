"""PyTorch forecasters sharing one monitored training loop.

Channel-independent models see one region at a time: input (B, L, C) -> (B, H),
with C = 1 value channel (+5 calendar channels if enabled), weights shared by all regions.
Graph models (T-GCN) see all regions jointly: (B, L, N, C) -> (B, H, N).
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
from torch import nn

from ..data import knn_adjacency
from .base import BaseModel, P, register

COMMON = [
    P("epochs", "训练轮数 epochs", "int", 30, min=1, max=1000),
    P("batch_size", "批大小 batch", "int", 256, min=8, max=8192),
    P("lr", "学习率", "float", 0.001, min=0.00001, max=1.0, step=0.0001),
    P("weight_decay", "权重衰减", "float", 0.0, min=0.0, max=0.1, step=0.0001),
    P("optimizer", "优化器", "select", "adam", options=["adam", "adamw", "rmsprop", "sgd"]),
    P("loss", "损失函数", "select", "mse", options=["mse", "mae", "huber"]),
    P("scheduler", "学习率调度", "select", "plateau", options=["none", "cosine", "step", "plateau"]),
    P("patience", "早停耐心（0=关闭）", "int", 8, min=0, max=200),
    P("grad_clip", "梯度裁剪（0=关闭）", "float", 5.0, min=0.0, max=100, step=0.5),
    P("epoch_samples", "每轮最多样本（0=全部）", "int", 0, min=0, max=5000000, step=1000),
    P("use_time_features", "加入时间特征", "bool", True),
    P("device", "计算设备", "select", "cpu", options=["cpu", "mps", "cuda", "auto"]),
    P("seed", "随机种子", "int", 42, min=0, max=99999),
]


def pick_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


# ----------------------------------------------------------------------------- networks
class MLPNet(nn.Module):
    def __init__(self, L, C, H, hidden, layers, dropout, act):
        super().__init__()
        A = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "silu": nn.SiLU}[act]
        mods, d = [nn.Flatten()], L * C
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), A(), nn.Dropout(dropout)]
            d = hidden
        mods.append(nn.Linear(d, H))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


class RNNNet(nn.Module):
    def __init__(self, cell, C, H, hidden, layers, dropout, bidirectional):
        super().__init__()
        R = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = R(C, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0,
                     bidirectional=bidirectional)
        self.bi = bidirectional
        self.head = nn.Linear(hidden * (2 if bidirectional else 1), H)

    def forward(self, x):
        _, h = self.rnn(x)
        h = h[0] if isinstance(h, tuple) else h
        z = torch.cat([h[-2], h[-1]], -1) if self.bi else h[-1]
        return self.head(z)


class Seq2SeqNet(nn.Module):
    uses_target = True

    def __init__(self, C, H, hidden, layers, dropout, teacher_forcing):
        super().__init__()
        self.H, self.tf_ratio = H, teacher_forcing
        self.enc = nn.LSTM(C, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0)
        self.dec = nn.LSTM(1, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, y=None):
        _, state = self.enc(x)
        inp = x[:, -1:, :1]
        outs = []
        for h in range(self.H):
            o, state = self.dec(inp, state)
            step = self.out(o)  # (B,1,1)
            outs.append(step[:, 0, 0])
            if self.training and y is not None and torch.rand(1).item() < self.tf_ratio:
                inp = y[:, h:h + 1, None]
            else:
                inp = step
        return torch.stack(outs, 1)


class CausalConv(nn.Module):
    """Dilated causal 1-D convolution as im2col + matmul (much faster than nn.Conv1d backward on CPU)."""

    def __init__(self, cin, cout, k, dil):
        super().__init__()
        self.k, self.dil = k, dil
        self.lin = nn.Linear(cin * k, cout)

    def forward(self, x):  # (B, C, T) -> (B, cout, T)
        span = (self.k - 1) * self.dil
        xp = nn.functional.pad(x, (span, 0))
        cols = xp.unfold(2, span + 1, 1)[..., ::self.dil]  # (B, C, T, k)
        B, C, T, k = cols.shape
        return self.lin(cols.permute(0, 2, 1, 3).reshape(B, T, C * k)).transpose(1, 2)


class TCNBlock(nn.Module):
    def __init__(self, cin, cout, k, dil, dropout):
        super().__init__()
        self.c1 = CausalConv(cin, cout, k, dil)
        self.c2 = CausalConv(cout, cout, k, dil)
        self.drop = nn.Dropout(dropout)
        self.res = CausalConv(cin, cout, 1, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        y = self.drop(torch.relu(self.c1(x)))
        y = self.drop(torch.relu(self.c2(y)))
        return torch.relu(y + self.res(x))


class TCNNet(nn.Module):
    def __init__(self, C, H, channels, levels, kernel, dropout):
        super().__init__()
        blocks, cin = [], C
        for i in range(levels):
            blocks.append(TCNBlock(cin, channels, kernel, 2 ** i, dropout))
            cin = channels
        self.net = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, H)
        self.receptive = 1 + 2 * (kernel - 1) * (2 ** levels - 1)

    def forward(self, x):
        return self.head(self.net(x.transpose(1, 2))[:, :, -1])


class TransformerNet(nn.Module):
    def __init__(self, L, C, H, d_model, nhead, layers, ff, dropout):
        super().__init__()
        self.inp = nn.Linear(C, d_model)
        self.pos = nn.Parameter(torch.randn(1, L, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(2 * d_model, H)

    def forward(self, x):
        z = self.enc(self.inp(x) + self.pos)
        return self.head(torch.cat([z[:, -1], z.mean(1)], -1))


class DLinearNet(nn.Module):
    def __init__(self, L, H, kernel):
        super().__init__()
        self.k = kernel if kernel % 2 == 1 else kernel + 1
        self.trend = nn.Linear(L, H)
        self.season = nn.Linear(L, H)

    def forward(self, x):
        v = x[..., 0]
        pad = (self.k - 1) // 2
        vp = torch.cat([v[:, :1].repeat(1, pad), v, v[:, -1:].repeat(1, pad)], 1)
        tr = nn.functional.avg_pool1d(vp[:, None], self.k, stride=1)[:, 0]
        return self.trend(tr) + self.season(v - tr)


class NBeatsNet(nn.Module):
    def __init__(self, L, H, blocks, hidden, layers):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(blocks):
            fc, d = [], L
            for _ in range(layers):
                fc += [nn.Linear(d, hidden), nn.ReLU()]
                d = hidden
            self.blocks.append(nn.ModuleDict({"fc": nn.Sequential(*fc), "b": nn.Linear(hidden, L),
                                              "f": nn.Linear(hidden, H)}))

    def forward(self, x):
        res = x[..., 0]
        fc = 0
        for blk in self.blocks:
            h = blk["fc"](res)
            res = res - blk["b"](h)
            fc = fc + blk["f"](h)
        return fc


class TGCNCell(nn.Module):
    def __init__(self, A, din, hidden):
        super().__init__()
        self.register_buffer("A", A)
        self.gates = nn.Linear(din + hidden, 2 * hidden)
        self.cand = nn.Linear(din + hidden, hidden)

    def gc(self, X):
        return torch.einsum("nm,bmf->bnf", self.A, X)

    def forward(self, x, h):
        r, u = torch.sigmoid(self.gates(self.gc(torch.cat([x, h], -1)))).chunk(2, -1)
        c = torch.tanh(self.cand(self.gc(torch.cat([x, r * h], -1))))
        return u * h + (1 - u) * c


class TGCNNet(nn.Module):
    graph = True

    def __init__(self, A, C, H, hidden, layers, dropout):
        super().__init__()
        self.cells = nn.ModuleList([TGCNCell(A, C if i == 0 else hidden, hidden) for i in range(layers)])
        self.hidden = hidden
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, H)

    def forward(self, x):  # (B, L, N, C)
        B, L, N, _ = x.shape
        hs = [x.new_zeros(B, N, self.hidden) for _ in self.cells]
        for t in range(L):
            inp = x[:, t]
            for i, cell in enumerate(self.cells):
                hs[i] = cell(inp, hs[i])
                inp = self.drop(hs[i])
        return self.head(inp).transpose(1, 2)  # (B, H, N)


# ----------------------------------------------------------------------------- base class
class TorchModel(BaseModel):
    category = "深度学习"
    graph = False

    @classmethod
    def all_params(cls):
        return list(cls.params) + COMMON

    def build(self, data, C) -> nn.Module:
        raise NotImplementedError

    # -- batching helpers (index-gather, no materialised window tensors) --
    def _prep(self, data):
        self.dev = pick_device(self.p["device"])
        self.S = torch.tensor(data.scaled, device=self.dev)
        self.TF = torch.tensor(data.tf, device=self.dev)
        self.arL = torch.arange(data.L, device=self.dev)
        self.arH = torch.arange(data.H, device=self.dev)

    def _units(self, data, starts):
        starts = torch.as_tensor(np.asarray(starts), device=self.dev, dtype=torch.long)
        if self.graph:
            return starts, None
        N = data.N
        return starts.repeat_interleave(N), torch.arange(N, device=self.dev).repeat(len(starts))

    def _batch(self, data, t, n):
        ti = t[:, None] + self.arL
        yi = t[:, None] + data.L + self.arH
        use_tf = self.p["use_time_features"]
        if self.graph:
            x = self.S[ti][..., None]  # (B,L,N,1)
            if use_tf:
                x = torch.cat([x, self.TF[ti][:, :, None, :].expand(-1, -1, data.N, -1)], -1)
            return x, self.S[yi]  # (B,H,N)
        x = self.S[ti, n[:, None]][..., None]  # (B,L,1)
        if use_tf:
            x = torch.cat([x, self.TF[ti]], -1)
        return x, self.S[yi, n[:, None]]  # (B,H)

    def _forward(self, x, y=None):
        if getattr(self.net, "uses_target", False):
            return self.net(x, y)
        return self.net(x)

    @torch.no_grad()
    def _predict_units(self, data, t, n, bs=4096):
        self.net.eval()
        out = []
        bs = max(8, bs // data.N) if self.graph else bs
        for i in range(0, len(t), bs):
            x, _ = self._batch(data, t[i:i + bs], None if n is None else n[i:i + bs])
            out.append(self._forward(x))
        return torch.cat(out)

    def predict(self, data, starts):
        t, n = self._units(data, starts)
        out = self._predict_units(data, t, n).float().cpu().numpy()
        if self.graph:
            return out
        return out.reshape(len(starts), data.N, data.H).transpose(0, 2, 1)

    def fit(self, data, mon):
        torch.manual_seed(self.p["seed"])
        np.random.seed(self.p["seed"])
        self._prep(data)
        C = 1 + (data.tf.shape[1] if self.p["use_time_features"] else 0)
        self.net = self.build(data, C).to(self.dev)
        nparams = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        mon.log(f"设备 {self.dev}，模型参数量 {nparams:,}")

        opt_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "rmsprop": torch.optim.RMSprop,
                   "sgd": lambda ps, **kw: torch.optim.SGD(ps, momentum=0.9, **kw)}[self.p["optimizer"]]
        opt = opt_cls(self.net.parameters(), lr=self.p["lr"], weight_decay=self.p["weight_decay"])
        E = self.p["epochs"]
        sched = {
            "none": None,
            "cosine": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=E),
            "step": torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, E // 3), gamma=0.3),
            "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3),
        }[self.p["scheduler"]]
        crit = {"mse": nn.MSELoss(), "mae": nn.L1Loss(), "huber": nn.HuberLoss(delta=1.0)}[self.p["loss"]]

        tr_t, tr_n = self._units(data, data.train_starts)
        vs = data.val_starts
        if len(vs) > 300:
            vs = vs[np.linspace(0, len(vs) - 1, 300).astype(int)]
        va_t, va_n = self._units(data, vs)
        yv_raw = data.Y(vs, scaled=False)
        yv = data.Y(vs)
        yv = torch.tensor(yv if self.graph else yv.transpose(0, 2, 1).reshape(-1, data.H), device=self.dev)
        U = len(tr_t)
        per_epoch = min(U, self.p["epoch_samples"]) if self.p["epoch_samples"] > 0 else U
        bs = self.p["batch_size"]
        nb = math.ceil(per_epoch / bs)
        mon.log(f"每轮 {per_epoch:,} 个训练样本，{nb} 个批次；验证窗口 {len(vs)}")

        best, best_state, bad = float("inf"), None, 0
        for ep in range(1, E + 1):
            self.net.train()
            perm = torch.randperm(U, device=self.dev)[:per_epoch]
            tot, t0 = 0.0, time.time()
            for b in range(nb):
                if mon.stopping:
                    break
                idx = perm[b * bs:(b + 1) * bs]
                x, y = self._batch(data, tr_t[idx], None if tr_n is None else tr_n[idx])
                opt.zero_grad()
                loss = crit(self._forward(x, y), y)
                loss.backward()
                if self.p["grad_clip"] > 0:
                    nn.utils.clip_grad_norm_(self.net.parameters(), self.p["grad_clip"])
                opt.step()
                tot += loss.item()
                mon.progress(((ep - 1) + (b + 1) / nb) / E, f"epoch {ep}/{E}，batch {b + 1}/{nb}")
            if mon.stopping:
                mon.log(f"第 {ep} 轮收到停止指令，恢复最佳权重")
                break
            pv = self._predict_units(data, va_t, va_n)
            vl = float(crit(pv, yv).item())
            pz = pv.float().cpu().numpy()
            pz = pz if self.graph else pz.reshape(len(vs), data.N, data.H).transpose(0, 2, 1)
            vmae = float(np.abs(data.inverse(pz) - yv_raw).mean())
            lr = opt.param_groups[0]["lr"]
            mon.curve(ep, train_loss=tot / max(nb, 1), val_loss=vl, val_mae=vmae, lr=lr)
            mon.log(f"epoch {ep:>3}  train={tot / max(nb, 1):.4f}  val={vl:.4f}  valMAE={vmae:,.2f}  "
                    f"lr={lr:.2e}  {time.time() - t0:.1f}s")
            if sched is not None:
                sched.step(vl) if self.p["scheduler"] == "plateau" else sched.step()
            if vl < best - 1e-6:
                best, bad = vl, 0
                best_state = copy.deepcopy(self.net.state_dict())
            else:
                bad += 1
                if self.p["patience"] and bad >= self.p["patience"]:
                    mon.log(f"验证损失 {bad} 轮未改善，早停（最佳 val={best:.4f}）")
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)


# ----------------------------------------------------------------------------- registered models
@register
class MLPModel(TorchModel):
    key, name = "mlp", "多层感知机 MLP"
    desc = "把输入窗口展平后经过若干全连接层直接输出 H 步预测。"
    params = [
        P("hidden", "隐藏单元", "int", 256, min=8, max=4096),
        P("layers", "隐藏层数", "int", 2, min=1, max=8),
        P("dropout", "Dropout", "float", 0.1, min=0, max=0.9, step=0.05),
        P("activation", "激活函数", "select", "relu", options=["relu", "gelu", "tanh", "silu"]),
    ]

    def build(self, data, C):
        return MLPNet(data.L, C, data.H, self.p["hidden"], self.p["layers"], self.p["dropout"], self.p["activation"])


@register
class LSTMModel(TorchModel):
    key, name = "lstm", "长短期记忆网络 LSTM"
    desc = "LSTM 编码输入序列，用最终隐藏状态经线性层输出全部预测步。"
    params = [
        P("hidden", "隐藏单元", "int", 64, min=4, max=1024),
        P("layers", "层数", "int", 2, min=1, max=6),
        P("dropout", "层间 Dropout", "float", 0.1, min=0, max=0.9, step=0.05),
        P("bidirectional", "双向", "bool", False),
    ]

    def build(self, data, C):
        return RNNNet("lstm", C, data.H, self.p["hidden"], self.p["layers"], self.p["dropout"], self.p["bidirectional"])


@register
class GRUModel(LSTMModel):
    key, name = "gru", "门控循环单元 GRU"
    desc = "GRU 编码输入序列（门控更少、更轻量），最终隐藏状态经线性层输出。"

    def build(self, data, C):
        return RNNNet("gru", C, data.H, self.p["hidden"], self.p["layers"], self.p["dropout"], self.p["bidirectional"])


@register
class Seq2SeqModel(TorchModel):
    key, name = "seq2seq", "编码器-解码器 Seq2Seq"
    desc = "LSTM 编码器 + 自回归 LSTM 解码器，训练时以一定概率使用 teacher forcing。"
    params = [
        P("hidden", "隐藏单元", "int", 64, min=4, max=1024),
        P("layers", "层数", "int", 1, min=1, max=4),
        P("dropout", "层间 Dropout", "float", 0.0, min=0, max=0.9, step=0.05),
        P("teacher_forcing", "Teacher forcing 比例", "float", 0.5, min=0, max=1, step=0.05),
    ]

    def build(self, data, C):
        return Seq2SeqNet(C, data.H, self.p["hidden"], self.p["layers"], self.p["dropout"], self.p["teacher_forcing"])


@register
class TCNModel(TorchModel):
    key, name = "tcn", "时间卷积网络 TCN"
    desc = "因果膨胀卷积残差块堆叠，感受野随层数指数增长。"
    params = [
        P("channels", "通道数", "int", 32, min=4, max=512),
        P("levels", "残差块层数", "int", 4, min=1, max=10),
        P("kernel", "卷积核大小", "int", 3, min=2, max=9),
        P("dropout", "Dropout", "float", 0.1, min=0, max=0.9, step=0.05),
    ]

    def build(self, data, C):
        net = TCNNet(C, data.H, self.p["channels"], self.p["levels"], self.p["kernel"], self.p["dropout"])
        return net


@register
class TransformerModel(TorchModel):
    key, name = "transformer", "Transformer"
    desc = "多头自注意力编码器 + 可学习位置编码，取末位与均值池化表征输出预测。"
    params = [
        P("d_model", "模型维度 d_model", "int", 64, min=8, max=512),
        P("nhead", "注意力头数", "int", 4, min=1, max=16),
        P("layers", "编码层数", "int", 2, min=1, max=8),
        P("ff", "前馈层维度", "int", 128, min=16, max=2048),
        P("dropout", "Dropout", "float", 0.1, min=0, max=0.9, step=0.05),
    ]

    def build(self, data, C):
        if self.p["d_model"] % self.p["nhead"]:
            raise ValueError("d_model 必须能被注意力头数整除")
        return TransformerNet(data.L, C, data.H, self.p["d_model"], self.p["nhead"], self.p["layers"],
                              self.p["ff"], self.p["dropout"])


@register
class DLinearModel(TorchModel):
    key, name = "dlinear", "DLinear"
    desc = "滑动平均把序列分解为趋势和季节两部分，各用一个线性层映射到未来（AAAI 2023）。"
    params = [P("kernel", "滑动平均窗口", "int", 25, min=3, max=301)]

    def build(self, data, C):
        return DLinearNet(data.L, data.H, min(self.p["kernel"], data.L if data.L % 2 else data.L - 1))


@register
class NBeatsModel(TorchModel):
    key, name = "nbeats", "N-BEATS"
    desc = "全连接块的双残差堆叠（backcast/forecast），通用架构版本（ICLR 2020）。"
    params = [
        P("blocks", "块数量", "int", 4, min=1, max=30),
        P("hidden", "隐藏单元", "int", 256, min=16, max=2048),
        P("layers", "每块全连接层数", "int", 3, min=1, max=8),
    ]

    def build(self, data, C):
        return NBeatsNet(data.L, data.H, self.p["blocks"], self.p["hidden"], self.p["layers"])


@register
class TGCNModel(TorchModel):
    key, name = "tgcn", "时空图卷积 T-GCN"
    category = "图神经网络"
    desc = ("图卷积嵌入 GRU 门控单元，同时建模区域间空间依赖与时间依赖。"
            "邻接图可按地理距离 kNN 或训练段流量相关性构建。")
    graph = True
    params = [
        P("hidden", "隐藏单元", "int", 64, min=4, max=512),
        P("layers", "层数", "int", 1, min=1, max=3),
        P("dropout", "Dropout", "float", 0.0, min=0, max=0.9, step=0.05),
        P("adjacency", "邻接图", "select", "distance", options=["distance", "correlation"]),
        P("k_neighbors", "近邻数 k", "int", 6, min=1, max=30),
    ]

    @classmethod
    def all_params(cls):
        ps = super().all_params()
        tuned = {"batch_size": 16, "lr": 0.005, "epochs": 60, "patience": 12}
        return [dict(p, default=tuned[p["name"]]) if p["name"] in tuned else p for p in ps]

    def build(self, data, C):
        k = self.p["k_neighbors"]
        if self.p["adjacency"] == "distance":
            A = knn_adjacency(data.coords, k)
        else:
            corr = np.nan_to_num(np.corrcoef(data.scaled[: data.train_end].T))
            np.fill_diagonal(corr, -np.inf)
            A = np.zeros_like(corr)
            idx = np.argsort(-corr, axis=1)[:, :k]
            A[np.repeat(np.arange(len(A)), k), idx.ravel()] = 1
            A = np.maximum(A, A.T)
        A = A + np.eye(len(A))
        d = A.sum(1) ** -0.5
        Ah = torch.tensor(d[:, None] * A * d[None, :], dtype=torch.float32)
        return TGCNNet(Ah, C, data.H, self.p["hidden"], self.p["layers"], self.p["dropout"])
