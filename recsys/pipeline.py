"""Orchestration : entraînement des deux étapes, inférence et évaluation de bout en bout."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .config import Config
from .data import Dataset, binarize, load_dataset, temporal_split, to_csr
from .metrics import catalog_coverage, evaluate_ranking, mean_log_popularity, summarize
from .ranking import FeatureContext, Ranker, build_features, labels_for, rerank
from .retrieval import FaissIndex, TwoTowerRetriever, exact_topk, generate_candidates


class TwoStageRecommender:
    def __init__(self, cfg: Config, data: Dataset):
        self.cfg, self.data = cfg, data

    def fit(self, splits):
        cfg, data = self.cfg, self.data
        nu, ni = data.n_users, data.n_items
        h_ret, h_rank, h_val = (to_csr(getattr(splits, s), nu, ni) for s in ("retriever", "ranker", "valid"))
        h_ret_rank = binarize(h_ret + h_rank)

        # --- étape 1 : two-tower + FAISS
        valid_users = np.unique(splits.valid["user"])
        self.retriever = TwoTowerRetriever(cfg, data)
        self.retriever.fit(splits.retriever, h_val, h_ret_rank, valid_users)
        self.index = FaissIndex(cfg.index_type).build(self.retriever.item_vectors())

        # popularité calculée sur le seul historique "retriever" : aucune fuite vers les labels du classeur
        pop = np.asarray(h_ret.sum(0)).ravel()
        self.ctx = FeatureContext.from_data(data, pop)

        # --- étape 2 : classeur entraîné sur les candidats du retriever
        rank_users = np.unique(splits.ranker["user"])
        f_tr, *_ = self._candidates(rank_users, h_ret)
        f_va, *_ = self._candidates(valid_users, h_ret_rank)
        y_tr, y_va = labels_for(f_tr, h_rank), labels_for(f_va, h_val)
        print(f"[ranker] train: {len(y_tr)} paires ({y_tr.mean():.2%} positives) | valid: {len(y_va)} ({y_va.mean():.2%})")
        self.ranker = Ranker(cfg, data)
        self.ranker.fit(f_tr, y_tr, f_va, y_va)

    def _candidates(self, user_ids, history):
        rows = history[user_ids]
        uv = self.retriever.user_vectors(user_ids)
        cand_ids, cand_scores = generate_candidates(self.index, uv, rows, self.cfg.n_candidates, self.cfg.overfetch)
        feats = build_features(user_ids, cand_ids, cand_scores, rows, self.ctx)
        return feats, cand_ids, cand_scores

    def recommend(self, user_ids, history, k):
        """Retourne les candidats, le top-k du retriever seul et le top-k reclassé."""
        feats, cand_ids, cand_scores = self._candidates(user_ids, history)
        ids, rank_scores, retr_scores = rerank(cand_ids, cand_scores, feats, self.ranker.predict(feats), k)
        return {
            "candidates": cand_ids,
            "retrieval": cand_ids[:, :k],
            "ranked": ids,
            "ranked_scores": rank_scores,
            "ranked_retrieval_scores": retr_scores,
        }

    def save(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.index.save(str(out_dir / "items.faiss"))
        self.retriever.model.save_weights(str(out_dir / "two_tower.weights.h5"))
        self.ranker.model.save(out_dir / "ranker.keras")


def run(cfg: Config) -> dict:
    tf.keras.utils.set_random_seed(cfg.seed)
    t0 = time.time()
    data = load_dataset(cfg.dataset, cfg.data_dir, cfg.seed)
    splits = temporal_split(data.interactions, cfg.positive_threshold, cfg.min_positives)
    nu, ni = data.n_users, data.n_items
    print(f"{data.name}: {nu} users, {ni} items, {len(data.interactions)} interactions | "
          f"retriever={len(splits.retriever)} ranker={len(splits.ranker)} valid={len(splits.valid)} test={len(splits.test)}")

    model = TwoStageRecommender(cfg, data)
    model.fit(splits)

    # --- évaluation finale : historique = tout ce qui précède le test
    h_ret, h_rank, h_val, h_test = (to_csr(getattr(splits, s), nu, ni) for s in ("retriever", "ranker", "valid", "test"))
    history = binarize(h_ret + h_rank + h_val)
    users = np.unique(splits.test["user"])
    kmax = max(max(cfg.ks), cfg.n_candidates)
    out = model.recommend(users, history, max(cfg.ks))

    pop_counts = np.asarray(history.sum(0)).ravel().astype(np.float32)
    _, pop_ids = exact_topk(np.ones((len(users), 1), np.float32), pop_counts[:, None], history[users], max(cfg.ks))

    runs = {
        "popularity": pop_ids,
        "retrieval_only": out["retrieval"],
        "two_stage": out["ranked"],
    }
    per_user = pd.concat([evaluate_ranking(r, users, h_test, cfg.ks, name) for name, r in runs.items()], ignore_index=True)
    ceiling = evaluate_ranking(out["candidates"], users, h_test, (cfg.n_candidates,), "candidate_ceiling")
    metrics = pd.concat([summarize(per_user), summarize(ceiling)], ignore_index=True)

    log_pop = np.log1p(pop_counts)
    diversity = pd.DataFrame([
        {"model": name, "k": 10, "catalog_coverage": catalog_coverage(r, ni, 10), "mean_log_popularity": mean_log_popularity(r, log_pop, 10)}
        for name, r in runs.items()
    ])
    print(f"Terminé en {time.time() - t0:.0f}s")
    return {
        "data": data, "splits": splits, "model": model, "users": users, "history": history,
        "runs": runs, "output": out, "per_user": per_user, "metrics": metrics, "diversity": diversity,
        "h_test": h_test,
    }
