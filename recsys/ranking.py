"""Étape 2 : features (utilisateur, item, candidat) et modèle de classement TensorFlow."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .data import Dataset


@dataclass
class FeatureContext:
    item_genres: np.ndarray  # (n_items, G)
    item_log_pop: np.ndarray  # (n_items,)
    item_year: np.ndarray  # (n_items,) normalisée

    @classmethod
    def from_data(cls, data: Dataset, pop_counts: np.ndarray) -> "FeatureContext":
        year = data.items["year"].to_numpy(np.float32)
        return cls(data.item_genres, np.log1p(pop_counts).astype(np.float32), (year - year.mean()) / (year.std() + 1e-6))


@dataclass
class Features:
    user: np.ndarray
    item: np.ndarray
    dense: np.ndarray
    row: np.ndarray  # position de l'utilisateur dans le lot
    col: np.ndarray  # position du candidat dans la liste (= rang du retriever)


def build_features(user_ids, cand_ids, cand_scores, hist_rows: sp.csr_matrix, ctx: FeatureContext) -> Features:
    """Une ligne par (utilisateur, candidat valide). `hist_rows` est aligné sur `user_ids`."""
    n, n_cand = cand_ids.shape
    valid = cand_ids >= 0
    safe = np.where(valid, cand_ids, 0)

    hist_len = np.asarray(hist_rows.sum(1)).ravel()
    denom = np.maximum(hist_len, 1)[:, None]
    profile = (hist_rows @ ctx.item_genres) / denom  # part de l'historique par genre
    user_mean_pop = (hist_rows @ ctx.item_log_pop)[:, None] / denom

    item_g = ctx.item_genres[safe]  # (n, N, G)
    affinity = np.einsum("ng,nkg->nk", profile, item_g) / np.maximum(item_g.sum(-1), 1)
    pop = ctx.item_log_pop[safe]
    scores = np.where(valid, cand_scores, 0.0)
    rank = np.broadcast_to(np.arange(n_cand), (n, n_cand))

    per_cand = np.stack([
        scores,
        scores - scores[:, :1],  # écart au meilleur candidat de l'utilisateur
        np.log1p(rank),
        pop,
        np.abs(pop - user_mean_pop),
        ctx.item_year[safe],
        affinity,
        np.broadcast_to(np.log1p(hist_len)[:, None], (n, n_cand)),
    ], axis=-1)
    dense = np.concatenate([per_cand, item_g], axis=-1)

    row, col = np.nonzero(valid)
    return Features(
        user=np.asarray(user_ids)[row].astype(np.int32),
        item=cand_ids[row, col].astype(np.int32),
        dense=dense[row, col].astype(np.float32),
        row=row, col=col,
    )


def labels_for(feats: Features, target: sp.csr_matrix) -> np.ndarray:
    return (np.asarray(target[feats.user, feats.item]).ravel() > 0).astype(np.float32)


class Ranker:
    """MLP pointwise : embeddings user/item + features denses -> probabilité de pertinence."""

    def __init__(self, cfg, data: Dataset):
        self.cfg, self.data = cfg, data
        self.mean = self.std = None
        self.model = None

    def _build(self, n_dense):
        cfg = self.cfg
        uid = keras.Input((), dtype="int32", name="user")
        iid = keras.Input((), dtype="int32", name="item")
        dense = keras.Input((n_dense,), name="dense")
        u = layers.Embedding(self.data.n_users, cfg.ranker_embedding_dim)(uid)
        i = layers.Embedding(self.data.n_items, cfg.ranker_embedding_dim)(iid)
        x = layers.Concatenate()([u, i, layers.Multiply()([u, i]), dense])
        for h in cfg.ranker_hidden:
            x = layers.Dense(h, activation="relu")(x)
            x = layers.Dropout(cfg.ranker_dropout)(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model([uid, iid, dense], out)
        model.compile(
            optimizer=keras.optimizers.Adam(cfg.ranker_lr),
            loss="binary_crossentropy",
            metrics=[keras.metrics.AUC(name="auc")],
        )
        return model

    def _inputs(self, f: Features):
        return {"user": f.user, "item": f.item, "dense": (f.dense - self.mean) / self.std}

    def fit(self, train: Features, y_train, valid: Features, y_valid, verbose=2):
        cfg = self.cfg
        self.mean = train.dense.mean(0)
        self.std = train.dense.std(0) + 1e-6
        self.model = self._build(train.dense.shape[1])
        cb = [keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=cfg.ranker_patience, restore_best_weights=True)]
        hist = self.model.fit(
            self._inputs(train), y_train,
            validation_data=(self._inputs(valid), y_valid),
            epochs=cfg.ranker_epochs, batch_size=cfg.ranker_batch_size,
            class_weight={0: 1.0, 1: cfg.ranker_pos_weight}, callbacks=cb, verbose=verbose,
        )
        return hist.history

    def predict(self, f: Features) -> np.ndarray:
        return self.model.predict(self._inputs(f), batch_size=8192, verbose=0).ravel()


def rerank(cand_ids, cand_scores, feats: Features, probs: np.ndarray, k: int):
    """Trie les candidats par probabilité du classeur ; rembourrage -1 / -inf."""
    mat = np.full(cand_ids.shape, -np.inf, dtype=np.float32)
    mat[feats.row, feats.col] = probs
    order = np.argsort(-mat, axis=1)[:, :k]
    top_scores = np.take_along_axis(mat, order, 1)
    top_ids = np.where(np.isfinite(top_scores), np.take_along_axis(cand_ids, order, 1), -1)
    top_retr = np.take_along_axis(cand_scores, order, 1)
    return top_ids, top_scores, top_retr
