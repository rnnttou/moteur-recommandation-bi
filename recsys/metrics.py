"""Métriques de classement : recall@k, NDCG@k, hit-rate@k (+ couverture / popularité pour le BI)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def hit_matrix(rec: np.ndarray, user_ids: np.ndarray, target: sp.csr_matrix) -> np.ndarray:
    """hits[i, j] = True si l'item recommandé en position j à l'utilisateur i est pertinent."""
    n, k = rec.shape
    rows = np.repeat(np.asarray(user_ids), k)
    cols = rec.ravel()
    valid = cols >= 0
    hits = np.zeros(n * k, dtype=bool)
    if valid.any():
        hits[valid] = np.asarray(target[rows[valid], cols[valid]]).ravel() > 0
    return hits.reshape(n, k)


def _pad(hits: np.ndarray, k: int) -> np.ndarray:
    if hits.shape[1] >= k:
        return hits[:, :k]
    return np.pad(hits, ((0, 0), (0, k - hits.shape[1])))


def recall_at_k(hits: np.ndarray, n_rel: np.ndarray, k: int) -> np.ndarray:
    return _pad(hits, k).sum(1) / n_rel


def hit_rate_at_k(hits: np.ndarray, k: int) -> np.ndarray:
    return _pad(hits, k).any(1).astype(float)


def ndcg_at_k(hits: np.ndarray, n_rel: np.ndarray, k: int) -> np.ndarray:
    """NDCG@k binaire : DCG = sum(rel_j / log2(j+2)), normalisé par le DCG idéal."""
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (_pad(hits, k) * disc).sum(1)
    ideal = np.concatenate([[0.0], np.cumsum(disc)])[np.minimum(n_rel, k)]
    return dcg / ideal


def evaluate_ranking(rec, user_ids, target, ks, model_name) -> pd.DataFrame:
    """Métriques par utilisateur (une ligne par user) pour un modèle."""
    user_ids = np.asarray(user_ids)
    n_rel = np.asarray(target[user_ids].sum(1)).ravel().astype(int)
    hits = hit_matrix(rec, user_ids, target)
    out = {"model": model_name, "user": user_ids, "n_test": n_rel}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(hits, n_rel, k)
        out[f"ndcg@{k}"] = ndcg_at_k(hits, n_rel, k)
        out[f"hit@{k}"] = hit_rate_at_k(hits, k)
    return pd.DataFrame(out)


def summarize(user_metrics: pd.DataFrame) -> pd.DataFrame:
    """Moyenne par modèle, au format long (model, metric, k, value) prêt pour un outil BI."""
    cols = [c for c in user_metrics.columns if "@" in c]
    wide = user_metrics.groupby("model", sort=False)[cols].mean().reset_index()
    long = wide.melt(id_vars="model", var_name="metric_k", value_name="value")
    long[["metric", "k"]] = long["metric_k"].str.split("@", expand=True)
    long["k"] = long["k"].astype(int)
    return long.drop(columns="metric_k")[["model", "metric", "k", "value"]]


def catalog_coverage(rec: np.ndarray, n_items: int, k: int) -> float:
    r = rec[:, :k]
    return len(np.unique(r[r >= 0])) / n_items


def mean_log_popularity(rec: np.ndarray, log_pop: np.ndarray, k: int) -> float:
    r = rec[:, :k]
    return float(log_pop[r[r >= 0]].mean())
