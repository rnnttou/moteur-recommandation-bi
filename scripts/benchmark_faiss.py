"""Compare les index FAISS (flat / HNSW / IVF) : recall vs. exact et latence, sur des embeddings synthétiques.

    python scripts/benchmark_faiss.py --n-items 200000 --dim 64
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recsys.retrieval import FaissIndex  # noqa: E402


def unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-items", type=int, default=100_000)
    p.add_argument("--n-queries", type=int, default=2000)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--k", type=int, default=100)
    a = p.parse_args()

    rng = np.random.default_rng(0)
    items, queries = unit(rng.normal(size=(a.n_items, a.dim))), unit(rng.normal(size=(a.n_queries, a.dim)))

    flat = FaissIndex("flat").build(items)
    t = time.perf_counter()
    _, truth = flat.search(queries, a.k)
    base = (time.perf_counter() - t) / a.n_queries * 1000
    print(f"{'index':8s} {'build(s)':>9s} {'ms/requête':>11s} {'recall@k vs exact':>18s}")
    print(f"{'flat':8s} {'-':>9s} {base:11.3f} {1.0:18.3f}")

    for kind in ("hnsw", "ivf"):
        t = time.perf_counter()
        idx = FaissIndex(kind).build(items)
        build = time.perf_counter() - t
        t = time.perf_counter()
        _, ids = idx.search(queries, a.k)
        ms = (time.perf_counter() - t) / a.n_queries * 1000
        recall = np.mean([len(np.intersect1d(ids[i], truth[i])) / a.k for i in range(a.n_queries)])
        print(f"{kind:8s} {build:9.1f} {ms:11.3f} {recall:18.3f}")


if __name__ == "__main__":
    main()
