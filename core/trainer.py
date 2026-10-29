"""Background training jobs with live event streams and on-disk persistence of results.

Each job runs in its own spawned subprocess. This keeps native thread pools
(MKL / OpenMP / PyTorch) of concurrent jobs and of the web server isolated —
mixing them in one process can deadlock — and lets "stop" fall back to a hard kill.
Events (logs, progress, curve points) flow back through a multiprocessing queue.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import threading
import time
import traceback
import uuid

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
MAX_WORKERS = 2
STOP_GRACE_SECONDS = 30
_ctx = mp.get_context("spawn")


class Job:
    def __init__(self, model_key, params, data):
        from .models import REGISTRY
        cls = REGISTRY[model_key]
        self.id = uuid.uuid4().hex[:10]
        self.model_key = model_key
        self.model_name = cls.name
        self.category = cls.category
        self.params = cls(params).p  # normalised + defaults filled in
        self.data_id = data.id
        self.data_name = data.cfg.name
        self.data_config = dict(vars(data.cfg))
        self.status = "queued"
        self.progress = 0.0
        self.events: list = []
        self.curve: list = []
        self.created = time.time()
        self.started = self.created
        self.finished = None
        self.stop_flag = False
        self.error = None
        self.metrics = None
        self.val_metrics = None
        self.train_time = None
        self.infer_time = None
        self._arrays = None  # lazily loaded result arrays

    def summary(self) -> dict:
        return {
            "id": self.id, "model_key": self.model_key, "model_name": self.model_name, "category": self.category,
            "params": self.params, "data_id": self.data_id, "data_name": self.data_name,
            "data_config": self.data_config, "status": self.status, "progress": self.progress,
            "created": self.created, "started": self.started, "finished": self.finished,
            "error": self.error, "metrics": self.metrics["overall"] if self.metrics else None,
            "val_metrics": self.val_metrics, "train_time": self.train_time, "infer_time": self.infer_time,
            "n_events": len(self.events),
        }

    def save_meta(self):
        meta = self.summary()
        meta.update({"metrics_full": self.metrics, "curve": self.curve, "events": self.events[-400:]})
        with open(os.path.join(RESULTS_DIR, f"{self.id}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, default=float)

    @property
    def arrays(self) -> dict:
        if self._arrays is None:
            path = os.path.join(RESULTS_DIR, f"{self.id}.npz")
            if not os.path.exists(path):
                raise KeyError("该任务没有可用的预测结果")
            z = np.load(path, allow_pickle=False)
            self._arrays = {k: z[k] for k in z.files}
        return self._arrays

    @classmethod
    def from_disk(cls, meta: dict) -> "Job":
        j = cls.__new__(cls)
        for k in ("id", "model_key", "model_name", "category", "params", "data_id", "data_name", "data_config",
                  "status", "progress", "created", "started", "finished", "error", "val_metrics",
                  "train_time", "infer_time"):
            setattr(j, k, meta.get(k))
        j.metrics = meta.get("metrics_full")
        j.curve = meta.get("curve", [])
        j.events = meta.get("events", [])
        j.stop_flag = False
        j._arrays = None
        return j


# ============================================================== child process side
class _QueueList:
    """list.append()-compatible sink that forwards to the parent."""

    def __init__(self, q, kind):
        self.q, self.kind = q, kind

    def append(self, item):
        self.q.put((self.kind, item))


class _ChildJob:
    def __init__(self, q, stop_event):
        self.events = _QueueList(q, "event")
        self.curve = _QueueList(q, "curve")
        self.started = time.time()
        self.progress = 0.0
        self._stop = stop_event

    @property
    def stop_flag(self):
        return self._stop.is_set()


def _child_main(job_id, model_key, params, data_config, q, stop_event):
    import warnings
    warnings.filterwarnings("ignore")
    from . import windows
    from .metrics import evaluate
    from .models import REGISTRY, Monitor, StopTraining

    try:  # share the CPU fairly between concurrent jobs
        import torch
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // MAX_WORKERS))
    except Exception:
        pass
    cj = _ChildJob(q, stop_event)
    mon = Monitor(cj)
    try:
        data = windows.WindowData(windows.WindowConfig(**data_config))
        model = REGISTRY[model_key](params)
        mon.log(f"开始训练 {model.name}，数据集 {data_config.get('name')}（pid {os.getpid()}）")
        mon.log("参数: " + ", ".join(f"{k}={v}" for k, v in model.p.items()))
        stopped = False
        try:
            model.fit(data, mon)
        except StopTraining:
            stopped = True
            mon.log("训练被用户中止")
        stopped = stopped or stop_event.is_set()
        train_time = time.time() - cj.started
        mon.progress(1.0, "评估测试集", force=True)
        mon.log(f"训练耗时 {train_time:.1f}s，开始在测试集上推理")
        t0 = time.time()
        try:
            pz = model.predict(data, data.test_starts)
        except Exception:
            if stopped:  # e.g. ARIMA interrupted before every region was fitted
                q.put(("result", {"status": "stopped"}))
                return
            raise
        infer_time = time.time() - t0
        pred = data.inverse(np.asarray(pz, dtype=np.float64))
        truth = data.Y(data.test_starts, scaled=False)
        metrics = evaluate(truth, pred)
        val_metrics = None
        if len(data.val_starts):
            vz = model.predict(data, data.val_starts)
            val_metrics = evaluate(data.Y(data.val_starts, scaled=False),
                                   data.inverse(np.asarray(vz, dtype=np.float64)))["overall"]
        np.savez_compressed(
            os.path.join(RESULTS_DIR, f"{job_id}.npz"),
            pred=pred.astype(np.float32), truth=truth.astype(np.float32), starts=np.asarray(data.test_starts),
            raw=data.raw.astype(np.float32), times=np.array([str(t) for t in data.times], dtype=str),
            regions=np.array(data.regions, dtype=str), labels=np.array(data.labels, dtype=str),
            L=np.asarray(data.L), H=np.asarray(data.H))
        o = metrics["overall"]
        fmt = lambda v, f: "-" if v is None else format(v, f)
        mon.log(f"测试集: MAE={fmt(o['MAE'], ',.3f')}  RMSE={fmt(o['RMSE'], ',.3f')}  "
                f"MAPE={fmt(o['MAPE'], '.2f')}%  R2={fmt(o['R2'], '.4f')}")
        q.put(("result", {"status": "stopped" if stopped else "done", "metrics": metrics,
                          "val_metrics": val_metrics, "train_time": train_time, "infer_time": infer_time}))
    except Exception as e:
        q.put(("result", {"status": "error", "error": f"{type(e).__name__}: {e}",
                          "trace": traceback.format_exc()[-2000:]}))


# ============================================================== parent side
JOBS: dict = {}
_queue: "queue.Queue[Job]" = queue.Queue()
_started = False


def _emit(job, ev):
    ev.setdefault("ts", round(time.time() - job.started, 3))
    job.events.append(ev)


def _handle(job, kind, item):
    """Apply one message from the child. Returns the final result dict, if this was it."""
    if kind == "event":
        job.events.append(item)
        if item.get("type") == "progress":
            job.progress = item["value"]
    elif kind == "curve":
        job.curve.append(item)
    elif kind == "result":
        return item
    return None


def _run(job: Job):
    job.status, job.started = "running", time.time()
    q, stop_event = _ctx.Queue(), _ctx.Event()
    p = _ctx.Process(target=_child_main, daemon=True,
                     args=(job.id, job.model_key, job.params, job.data_config, q, stop_event))
    p.start()
    result, stop_at = None, None
    while True:
        try:
            kind, item = q.get(timeout=0.3)
            result = _handle(job, kind, item) or result
            continue
        except queue.Empty:
            pass
        if job.stop_flag and not stop_event.is_set():
            stop_event.set()
            stop_at = time.time()
        if stop_at and p.is_alive() and time.time() - stop_at > STOP_GRACE_SECONDS:
            p.terminate()
            _emit(job, {"type": "log", "msg": "任务未在宽限期内响应停止指令，已强制终止进程"})
        if not p.is_alive():
            while True:  # drain anything left in the pipe
                try:
                    kind, item = q.get(timeout=0.5)
                    result = _handle(job, kind, item) or result
                except queue.Empty:
                    break
            break
    p.join(timeout=5)
    # fill in results before flipping status, so pollers never see "done" without metrics
    if result is None:
        status = "stopped" if job.stop_flag else "error"
        if status == "error":
            job.error = f"训练进程异常退出（exit code {p.exitcode}）"
            _emit(job, {"type": "error", "msg": job.error})
    else:
        status = result["status"]
        job.metrics = result.get("metrics")
        job.val_metrics = result.get("val_metrics")
        job.train_time = result.get("train_time")
        job.infer_time = result.get("infer_time")
        if status == "error":
            job.error = result["error"]
            _emit(job, {"type": "error", "msg": result["error"], "trace": result.get("trace", "")})
    job.finished = time.time()
    if status in ("done", "stopped"):
        job.progress = 1.0
    job.status = status
    job.save_meta()
    _emit(job, {"type": "done", "status": job.status, "metrics": job.metrics["overall"] if job.metrics else None})


def _worker():
    while True:
        job = _queue.get()
        if job.stop_flag:
            job.status, job.finished = "stopped", time.time()
            _emit(job, {"type": "done", "status": "stopped", "metrics": None})
            continue
        try:
            _run(job)
        except Exception as e:  # never let a worker thread die
            job.status, job.error = "error", f"{type(e).__name__}: {e}"
            _emit(job, {"type": "done", "status": "error", "metrics": None})


def start_workers():
    global _started
    if not _started:
        _started = True
        for _ in range(MAX_WORKERS):
            threading.Thread(target=_worker, daemon=True).start()


def submit(model_key: str, params: dict, data) -> Job:
    start_workers()
    job = Job(model_key, params, data)
    JOBS[job.id] = job
    _queue.put(job)
    return job


def load_saved():
    for fn in sorted(os.listdir(RESULTS_DIR)):
        if fn.endswith(".json"):
            try:
                with open(os.path.join(RESULTS_DIR, fn), encoding="utf-8") as f:
                    meta = json.load(f)
                if meta["id"] not in JOBS and meta.get("status") in ("done", "stopped", "error"):
                    JOBS[meta["id"]] = Job.from_disk(meta)
            except Exception:
                pass


def delete(job_id: str):
    job = JOBS.pop(job_id, None)
    if job:
        job.stop_flag = True
    for ext in (".json", ".npz"):
        p = os.path.join(RESULTS_DIR, f"{job_id}{ext}")
        if os.path.exists(p):
            os.remove(p)
