"""智能流量分析平台 — FastAPI backend.

Run:  python app.py   (then open http://127.0.0.1:8000)
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import re
import warnings
from functools import lru_cache

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

warnings.filterwarnings("ignore")

from core import cluster, data, eda, trainer, windows  # noqa: E402
from core.metrics import METRIC_INFO, evaluate  # noqa: E402
from core.models import catalog  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="智能流量分析平台")
_NONFINITE = re.compile(r"\bNaN\b|-?\bInfinity\b")


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def J(obj) -> Response:
    """JSON response that tolerates numpy types and maps NaN/Inf to null."""
    s = json.dumps(obj, ensure_ascii=False, default=_default, allow_nan=True)
    return Response(_NONFINITE.sub("null", s), media_type="application/json")


def _ds(key: str):
    if key not in data.DATASETS:
        raise HTTPException(404, f"未知数据集 {key}")
    return data.load(key)


# ------------------------------------------------------------------ datasets & EDA
@app.get("/api/datasets")
def datasets():
    return J([{"key": k, "title": v["title"]} for k, v in data.DATASETS.items()])


@app.get("/api/datasets/{key}")
def dataset_meta(key: str):
    return J(_ds(key).meta())


@app.get("/api/geo/{key}")
def geo(key: str):
    return J(_ds(key).geo)


@lru_cache(maxsize=16)
def _overview(key):
    return eda.overview(_ds(key))


@lru_cache(maxsize=16)
def _spatial(key, knn):
    return eda.spatial(_ds(key), knn)


@app.get("/api/eda/overview/{key}")
def eda_overview(key: str):
    return J(_overview(key))


@app.get("/api/eda/temporal/{key}")
def eda_temporal(key: str, regions: str = "", freq: str = "1h"):
    rs = [r for r in regions.split("|") if r]
    return J(eda.temporal(_ds(key), rs, freq))


@app.get("/api/eda/spatial/{key}")
def eda_spatial(key: str, knn: int = 6):
    return J(_spatial(key, knn))


@app.get("/api/eda/snapshot/{key}")
def eda_snapshot(key: str, i: int = 0, freq: str = "1h"):
    return J(eda.snapshot(_ds(key), freq, i))


# ------------------------------------------------------------------ clustering
@app.get("/api/cluster/methods")
def cluster_methods():
    return J({"methods": cluster.METHODS, "features": cluster.FEATURES, "norms": cluster.NORMS})


@app.post("/api/cluster/run")
def cluster_run(body: dict = Body(...)):
    try:
        return J(cluster.run(_ds(body["dataset"]), body.get("method", "kmeans"), body.get("feature", "wd_we"),
                             body.get("norm", "shape"), body.get("params", {}), body.get("embed", "pca")))
    except Exception as e:
        raise HTTPException(400, f"{type(e).__name__}: {e}")


@app.post("/api/cluster/elbow")
def cluster_elbow(body: dict = Body(...)):
    return J(cluster.elbow(_ds(body["dataset"]), body.get("feature", "wd_we"), body.get("norm", "shape"),
                           int(body.get("kmax", 10))))


# ------------------------------------------------------------------ forecasting datasets
@app.post("/api/windows")
def build_windows(body: dict = Body(...)):
    try:
        wd = windows.build(body)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))
    return J(wd.summary())


@app.get("/api/windows")
def list_windows():
    return J([wd.summary() | {"preview": None, "example": None, "regions": None}
              for wd in sorted(windows.DATASTORE.values(), key=lambda w: -w.created)])


@app.get("/api/windows/{wid}")
def get_window(wid: str):
    if wid not in windows.DATASTORE:
        raise HTTPException(404, "数据集不存在（服务重启后需重新构建）")
    return J(windows.DATASTORE[wid].summary())


@app.delete("/api/windows/{wid}")
def del_window(wid: str):
    windows.remove(wid)
    return J({"ok": True})


# ------------------------------------------------------------------ models & training
@app.get("/api/models")
def models():
    return J(catalog())


@app.post("/api/train")
def train(body: dict = Body(...)):
    wid = body.get("data_id")
    if wid not in windows.DATASTORE:
        raise HTTPException(400, "请先在「预测数据构建」面板构建数据集")
    # body: {data_id, models: [key, ...], params: {key: {param: value}}}
    params = body.get("params") or {}
    jobs = [trainer.submit(k, params.get(k, {}), windows.DATASTORE[wid]) for k in body.get("models", [])]
    return J([j.summary() for j in jobs])


@app.get("/api/jobs")
def jobs():
    return J([j.summary() for j in sorted(trainer.JOBS.values(), key=lambda j: -j.created)])


def _job(jid):
    if jid not in trainer.JOBS:
        raise HTTPException(404, "任务不存在")
    return trainer.JOBS[jid]


@app.get("/api/jobs/{jid}")
def job(jid: str):
    j = _job(jid)
    return J(j.summary() | {"curve": j.curve, "events": j.events[-300:]})


@app.post("/api/jobs/{jid}/stop")
def stop(jid: str):
    _job(jid).stop_flag = True
    return J({"ok": True})


@app.delete("/api/jobs/{jid}")
def delete_job(jid: str):
    trainer.delete(jid)
    return J({"ok": True})


@app.get("/api/jobs/{jid}/stream")
async def stream(jid: str, since: int = 0):
    j = _job(jid)

    async def gen():
        i, idle = since, 0
        while True:
            evs = j.events[i:]
            if evs:
                i += len(evs)
                idle = 0
                yield "data: " + _NONFINITE.sub("null", json.dumps(
                    {"events": evs, "status": j.status, "progress": j.progress, "next": i},
                    ensure_ascii=False, default=_default)) + "\n\n"
            elif j.status not in ("queued", "running") and (not j.events or j.events[-1].get("type") == "done"):
                yield "event: end\ndata: {}\n\n"
                return
            else:
                idle += 1
                if idle % 50 == 0:
                    yield ": keep-alive\n\n"
            await asyncio.sleep(0.25)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ------------------------------------------------------------------ evaluation
def _arrays(jid):
    j = _job(jid)
    try:
        return j, j.arrays
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.get("/api/metrics/info")
def metrics_info():
    return J(METRIC_INFO)


@app.get("/api/eval/{jid}/overview")
def eval_overview(jid: str):
    j, a = _arrays(jid)
    pred, truth = a["pred"], a["truth"]
    res = (pred - truth).ravel()
    lo, hi = np.percentile(res, [0.5, 99.5])
    hist, edges = np.histogram(np.clip(res, lo, hi), bins=60)
    rng = np.random.default_rng(0)
    k = min(4000, truth.size)
    sel = rng.choice(truth.size, k, replace=False)
    times = a["times"]
    L = int(a["L"])
    starts = a["starts"]
    return J({
        "job": j.summary(),
        "metrics": j.metrics,
        "regions": [{"id": r, "label": l} for r, l in zip(a["regions"].tolist(), a["labels"].tolist())],
        "dataset": j.data_config.get("dataset"),
        "H": int(a["H"]), "L": L, "n_test": int(len(starts)),
        "origins": [str(times[s + L - 1]) for s in starts],
        "residual_hist": {"counts": hist.tolist(), "edges": np.round(edges, 3).tolist()},
        "scatter": np.round(np.c_[truth.ravel()[sel], pred.ravel()[sel]], 2).tolist(),
    })


@app.get("/api/eval/{jid}/series")
def eval_series(jid: str, region: int = -1, h: int = 1):
    """Truth vs prediction along the test period for a fixed horizon step h (1-based)."""
    _, a = _arrays(jid)
    pred, truth, starts, times = a["pred"], a["truth"], a["starts"], a["times"]
    L, H = int(a["L"]), int(a["H"])
    h = int(np.clip(h, 1, H)) - 1
    if region < 0:
        y, p = truth[:, h].sum(1), pred[:, h].sum(1)
    else:
        y, p = truth[:, h, region], pred[:, h, region]
    return J({"t": [str(times[s + L + h]) for s in starts], "y": np.round(y, 2), "p": np.round(p, 2),
              "metrics": evaluate(y[:, None, None], p[:, None, None])["overall"]})


@app.get("/api/eval/{jid}/sample")
def eval_sample(jid: str, i: int = 0, region: int = -1):
    _, a = _arrays(jid)
    pred, truth, starts, times, raw = a["pred"], a["truth"], a["starts"], a["times"], a["raw"]
    L, H = int(a["L"]), int(a["H"])
    i = int(np.clip(i, 0, len(starts) - 1))
    s = int(starts[i])
    sel = (lambda m: m.sum(-1)) if region < 0 else (lambda m: m[..., region])
    return J({
        "i": i, "t_in": times[s:s + L].tolist(), "t_out": times[s + L:s + L + H].tolist(),
        "x": np.round(sel(raw[s:s + L]), 2), "y": np.round(sel(truth[i]), 2), "p": np.round(sel(pred[i]), 2),
    })


@app.get("/api/eval/compare")
def eval_compare(ids: str = Query("")):
    out = []
    for jid in [x for x in ids.split(",") if x]:
        j = trainer.JOBS.get(jid)
        if j and j.metrics:
            out.append({"id": jid, "model_name": j.model_name, "category": j.category, "data_name": j.data_name,
                        "data_id": j.data_id, "train_time": j.train_time, "infer_time": j.infer_time,
                        "status": j.status, "overall": j.metrics["overall"],
                        "per_horizon": j.metrics["per_horizon"], "val": j.val_metrics})
    return J(out)


@app.get("/api/eval/{jid}/export")
def export(jid: str, kind: str = "regions"):
    j, a = _arrays(jid)
    buf = io.StringIO()
    if kind == "regions":
        pr = j.metrics["per_region"]
        keys = list(pr)
        buf.write("region,label," + ",".join(keys) + "\n")
        for n, (r, l) in enumerate(zip(a["regions"].tolist(), a["labels"].tolist())):
            buf.write(f"{r},\"{l}\"," + ",".join("" if pr[k][n] is None else f"{pr[k][n]:.6g}" for k in keys) + "\n")
    else:  # predictions, long format
        L, H = int(a["L"]), int(a["H"])
        times, labels = a["times"], a["labels"].tolist()
        buf.write("origin,target_time,horizon,region,truth,pred\n")
        for i, s in enumerate(a["starts"]):
            for h in range(H):
                tt = times[s + L + h]
                for n, lab in enumerate(labels):
                    buf.write(f"{times[s + L - 1]},{tt},{h + 1},\"{lab}\",{a['truth'][i, h, n]:.4f},{a['pred'][i, h, n]:.4f}\n")
    name = f"{j.model_key}_{jid}_{kind}.csv"
    return Response("﻿" + buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


# ------------------------------------------------------------------ read-only demo mode
# Visitors arriving through a public tunnel (Cloudflare adds cf-connecting-ip) may browse everything
# but cannot change shared state. Local use on localhost keeps full access; TL_READONLY=1 forces read-only.
_WRITE_ROUTES = [
    ("POST", re.compile(r"^/api/windows$")), ("DELETE", re.compile(r"^/api/windows/[^/]+$")),
    ("POST", re.compile(r"^/api/train$")), ("POST", re.compile(r"^/api/jobs/[^/]+/stop$")),
    ("DELETE", re.compile(r"^/api/jobs/[^/]+$")),
]


def is_readonly(request) -> bool:
    if os.environ.get("TL_READONLY") == "1":
        return True
    if "cf-connecting-ip" in request.headers or "x-forwarded-for" in request.headers:
        return True
    host = request.headers.get("host", "").rsplit(":", 1)[0].strip("[]")
    return host not in ("127.0.0.1", "localhost", "::1")


@app.middleware("http")
async def readonly_guard(request, call_next):
    if any(request.method == m and rx.match(request.url.path) for m, rx in _WRITE_ROUTES) and is_readonly(request):
        return Response(json.dumps({"detail": "当前是只读演示模式，不能提交训练、删除结果或构建数据集。"}, ensure_ascii=False),
                        status_code=403, media_type="application/json")
    return await call_next(request)


@app.get("/api/mode")
def mode(request: Request):
    return {"readonly": is_readonly(request)}


# ------------------------------------------------------------------ static frontend
app.mount("/static", StaticFiles(directory=os.path.join(BASE, "static")), name="static")


@app.middleware("http")
async def revalidate_static(request, call_next):
    resp = await call_next(request)
    if request.url.path == "/" or (request.url.path.startswith("/static/") and "/vendor/" not in request.url.path):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.get("/")
def index():
    return FileResponse(os.path.join(BASE, "static", "index.html"))


@app.on_event("startup")
def _restore_state():
    trainer.load_saved()
    windows.load_saved()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
