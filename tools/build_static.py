"""Export a read-only static copy of TrafficLens into docs/ for GitHub Pages.

Every GET the frontend makes is precomputed into docs/api/**.json. Clustering is precomputed on a parameter
grid, EDA temporal analysis per region (combined in the browser), and evaluation series/samples are computed
in the browser from compact binary arrays. Training and dataset building are not available in the static copy.

Run:  python tools/build_static.py            (takes a few minutes; writes docs/)
"""
from __future__ import annotations

import functools
import itertools
import json
import os
import shutil
import sys
import time

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)

from fastapi.testclient import TestClient  # noqa: E402

import app as server  # noqa: E402
from core import cluster, data, eda  # noqa: E402

OUT = os.path.join(BASE, "docs")
DATASETS = ["milano", "trentino"]
STATIC_FREQS = ["1h", "2h", "4h", "1D"]  # coarser series are summed from hourly values in the browser

# Clustering grid: every select option, a few values for each numeric parameter.
GRID = {
    "k": [2, 3, 4, 5, 6, 7, 8], "n_init": [10], "n_neighbors": [5, 10, 20],
    "eps": [0.35, 0.5, 1, 2, 4, 8], "min_samples": [3, 5], "min_cluster_size": [3, 5, 8, 12],
}


def write(rel, payload):
    path = os.path.join(OUT, "api", rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(payload, (bytes, bytearray)):
        mode, data_ = "wb", payload
    elif isinstance(payload, str):
        mode, data_ = "w", payload
    else:
        mode, data_ = "w", server.J(payload).body.decode()
    with open(path, mode, **({} if mode == "wb" else {"encoding": "utf-8"})) as f:
        f.write(data_)


def get(client, url):
    r = client.get(url)
    r.raise_for_status()
    return r.json()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ frontend files
def copy_frontend():
    if os.path.exists(os.path.join(OUT, "static")):
        shutil.rmtree(os.path.join(OUT, "static"))
    shutil.copytree(os.path.join(BASE, "static"), os.path.join(OUT, "static"))
    os.remove(os.path.join(OUT, "static", "index.html"))
    html = open(os.path.join(BASE, "static", "index.html"), encoding="utf-8").read()
    flag = '<script>window.TL_STATIC = true;</script>\n<script src="static/js/common.js"></script>'
    assert '<script src="static/js/common.js"></script>' in html
    html = html.replace('<script src="static/js/common.js"></script>', flag)
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    open(os.path.join(OUT, ".nojekyll"), "w").close()


# ------------------------------------------------------------------ datasets, EDA
def export_eda(client):
    write("datasets.json", [d for d in get(client, "/api/datasets") if d["key"] in DATASETS])
    for key in DATASETS:
        meta = get(client, f"/api/datasets/{key}")
        meta["freq_options"] = [f for f in STATIC_FREQS if f in meta["freq_options"]]
        write(f"datasets/{key}.json", meta)
        write(f"geo/{key}.json", get(client, f"/api/geo/{key}"))
        write(f"eda/overview/{key}.json", get(client, f"/api/eda/overview/{key}"))
        write(f"eda/spatial/{key}.json", get(client, f"/api/eda/spatial/{key}"))

        ds = data.load(key)
        ids = list(ds.regions)
        index = None
        for i, rid in enumerate(ids):
            d = eda.temporal(ds, [rid], "1h")
            if index is None:
                index = {"ids": ids, "pos": {r: n for n, r in enumerate(ids)}, "t": d["series"]["t"],
                         "daily_t": d["daily"]["t"], "offday": d["daily"]["offday"]}
            assert d["series"]["t"] == index["t"]
            write(f"eda/temporal/{key}/r/{i}.json", {
                "label": d["regions"][0]["label"], "h": d["series"]["v"][rid], "profiles": d["profiles"][rid],
                "acf": d["acf"][rid], "daily": d["daily"]["v"][rid]})
            dc = d["decomp"]
            if dc and "t" in dc:
                assert dc["t"] == index["t"]
                dc = {k: v for k, v in dc.items() if k != "t"}
            write(f"eda/temporal/{key}/d/{i}.json", dc)
            if i % 25 == 0:
                log(f"  {key} temporal {i + 1}/{len(ids)}")
        write(f"eda/temporal/{key}/index.json", index)
        log(f"EDA {key} done")


# ------------------------------------------------------------------ clustering
def grid_params(spec):
    names, choices = [], []
    for p in spec["params"]:
        names.append(p["name"])
        choices.append(p["options"] if p["type"] == "select" else GRID[p["name"]])
    for combo in itertools.product(*choices):
        yield dict(zip(names, combo))


def pkey(params):
    return "_".join(f"{k}-{params[k]}" for k in sorted(params))


def export_cluster(client):
    methods = get(client, "/api/cluster/methods")
    for m in methods["methods"].values():  # numeric inputs become selects over the precomputed grid
        for p in m["params"]:
            if p["type"] != "select":
                p.update(type="select", options=[str(v) for v in GRID[p["name"]]], default=str(p["default"]))
                assert p["default"] in p["options"], p
    write("cluster/methods.json", methods)

    # profiles and stats are recomputed on every call; cache them for the grid
    def by_dataset(fn):  # Dataset is unhashable, so key the cache on the object id
        memo = {}
        return functools.wraps(fn)(lambda ds: memo[id(ds)] if id(ds) in memo else memo.setdefault(id(ds), fn(ds)))
    cluster._profiles = by_dataset(cluster._profiles)
    cluster.region_stats = by_dataset(cluster.region_stats)
    combos = [(f, n) for f in cluster.FEATURES if f != "stats" for n in cluster.NORMS] + [("stats", "shape")]
    for key in DATASETS:
        ds = data.load(key)
        for feature, norm in combos:
            fk = "stats" if feature == "stats" else f"{feature}_{norm}"
            for embed in ("pca", "tsne"):
                r = cluster.run(ds, "kmeans", feature, norm, {"k": 5}, embed)
                write(f"cluster/embed/{key}/{fk}_{embed}.json",
                      {"embedding": r["embedding"], "explained_variance": r["explained_variance"]})
            write(f"cluster/elbow/{key}/{fk}.json", cluster.elbow(ds, feature, norm, 10))
            n = 0
            for method, spec in cluster.METHODS.items():
                for params in grid_params(spec):
                    try:
                        r = cluster.run(ds, method, feature, norm, params, "pca")
                        r.pop("embedding"), r.pop("explained_variance")
                    except Exception as e:
                        r = {"__error": f"{type(e).__name__}: {e}"}
                    write(f"cluster/run/{key}/{fk}/{method}_{pkey(params)}.json", r)
                    n += 1
            log(f"  cluster {key} {fk}: {n} runs")


# ------------------------------------------------------------------ windows, training, evaluation
def export_results(client):
    wins = [w for w in get(client, "/api/windows") if w["config"]["dataset"] in DATASETS]
    write("windows.json", wins)
    for w in wins:
        write(f"windows/{w['id']}.json", get(client, f"/api/windows/{w['id']}"))
    write("models.json", get(client, "/api/models"))
    write("metrics/info.json", get(client, "/api/metrics/info"))

    jobs = [j for j in get(client, "/api/jobs") if (j.get("data_config") or {}).get("dataset") in DATASETS]
    write("jobs.json", jobs)
    write("eval/compare.json", {c["id"]: c for c in get(client, "/api/eval/compare?ids=" + ",".join(j["id"] for j in jobs))})
    done_windows = set()
    for j in jobs:
        jid = j["id"]
        write(f"jobs/{jid}.json", get(client, f"/api/jobs/{jid}"))
        if not j["metrics"]:
            continue
        write(f"eval/{jid}/overview.json", get(client, f"/api/eval/{jid}/overview"))
        r = client.get(f"/api/eval/{jid}/export?kind=regions")
        r.raise_for_status()
        write(f"eval/{jid}/regions.csv", r.content)

        a = server.trainer.JOBS[jid].arrays
        L, H, starts, raw = int(a["L"]), int(a["H"]), a["starts"], a["raw"].astype(np.float32)
        # truth is a slice of raw, so the browser only needs raw once per window
        idx = starts[:, None] + L + np.arange(H)[None, :]
        assert np.allclose(raw[idx], a["truth"], rtol=1e-5, atol=1e-3), jid
        wid = j["data_id"]
        if wid not in done_windows:
            done_windows.add(wid)
            write(f"eval/win/{wid}/meta.json", {"L": L, "H": H, "N": int(raw.shape[1]), "starts": starts.tolist(),
                                                "times": a["times"].tolist()})
            write(f"eval/win/{wid}/raw.bin", raw.astype("<f4").tobytes())
        # predictions as int16 with a per-region scale (error below 1/65534 of each region's range)
        pred = a["pred"].astype(np.float64)
        scale = np.maximum(np.abs(pred).max(axis=(0, 1)), 1e-9) / 32767
        q = np.round(pred / scale).astype("<i2")
        write(f"eval/{jid}/pred.bin", q.tobytes())
        write(f"eval/{jid}/arrays.json", {"wid": wid, "scale": scale.tolist()})
    log(f"results: {len(wins)} windows, {len(jobs)} jobs")


if __name__ == "__main__":
    # optional arguments pick stages, e.g. `frontend` alone re-copies the HTML/JS without recomputing
    only = set(sys.argv[1:]) or {"frontend", "eda", "cluster", "results"}
    if len(only) == 4 and os.path.exists(OUT):
        shutil.rmtree(OUT)
    if "frontend" in only:
        copy_frontend()
    with TestClient(server.app) as client:
        if "results" in only:
            export_results(client)
        if "eda" in only:
            export_eda(client)
        if "cluster" in only:
            export_cluster(client)
    log("done:", OUT)
