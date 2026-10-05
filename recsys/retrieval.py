"""Étape 1 : modèle two-tower (TensorFlow) + index FAISS pour la génération de candidats."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

try:
    import faiss
except Exception as exc:  # DLL bloquée (ex. Smart App Control sous Windows) ou paquet absent
    faiss = None
    print(f"[retrieval] FAISS indisponible ({type(exc).__name__}) -> repli sur un index NumPy exact.")

from .data import Dataset
from .metrics import hit_matrix, ndcg_at_k, recall_at_k


# --------------------------------------------------------------------------- modèle
class TwoTowerModel(keras.Model):
    """Deux tours MLP produisant des embeddings L2-normalisés (produit scalaire = cosinus)."""

    def __init__(self, n_users, n_items, n_genders, n_ages, n_occs, n_genres, dim, hidden):
        super().__init__()
        self.user_emb = layers.Embedding(n_users, dim)
        self.gender_emb = layers.Embedding(n_genders, 8)
        self.age_emb = layers.Embedding(n_ages, 8)
        self.occ_emb = layers.Embedding(n_occs, 8)
        self.user_mlp = keras.Sequential([layers.Dense(hidden, activation="relu"), layers.Dense(dim)])
        self.item_emb = layers.Embedding(n_items, dim)
        self.item_mlp = keras.Sequential([layers.Dense(hidden, activation="relu"), layers.Dense(dim)])

    def user_vec(self, uid, gender, age, occ):
        x = tf.concat([self.user_emb(uid), self.gender_emb(gender), self.age_emb(age), self.occ_emb(occ)], -1)
        return tf.math.l2_normalize(self.user_mlp(x), axis=-1)

    def item_vec(self, iid, genres):
        x = tf.concat([self.item_emb(iid), genres], -1)
        return tf.math.l2_normalize(self.item_mlp(x), axis=-1)


class TwoTowerRetriever:
    def __init__(self, cfg, data: Dataset):
        self.cfg, self.data = cfg, data
        u = data.users
        self._gender = tf.constant(u["gender_id"].to_numpy(), tf.int32)
        self._age = tf.constant(u["age_id"].to_numpy(), tf.int32)
        self._occ = tf.constant(u["occupation_id"].to_numpy(), tf.int32)
        self._genres = tf.constant(data.item_genres, tf.float32)
        self.model = TwoTowerModel(
            data.n_users, data.n_items, 2, int(u["age_id"].max()) + 1, int(u["occupation_id"].max()) + 1,
            len(data.genre_names), cfg.embedding_dim, cfg.hidden_dim,
        )
        self._build()

    def _build(self):
        z = tf.zeros([1], tf.int32)
        self._user_vec(z)
        self._item_vec(z)

    def _user_vec(self, uid):
        return self.model.user_vec(uid, tf.gather(self._gender, uid), tf.gather(self._age, uid), tf.gather(self._occ, uid))

    def _item_vec(self, iid):
        return self.model.item_vec(iid, tf.gather(self._genres, iid))

    # --- inférence
    def user_vectors(self, user_ids, batch=4096) -> np.ndarray:
        ids = np.asarray(user_ids, np.int32)
        return np.concatenate([self._user_vec(tf.constant(ids[i:i + batch])).numpy() for i in range(0, len(ids), batch)])

    def item_vectors(self, batch=4096) -> np.ndarray:
        ids = np.arange(self.data.n_items, dtype=np.int32)
        return np.concatenate([self._item_vec(tf.constant(ids[i:i + batch])).numpy() for i in range(0, len(ids), batch)])

    # --- entraînement : softmax avec négatifs in-batch + correction logQ
    def fit(self, train, valid_target, valid_seen, valid_users, verbose=True):
        cfg = self.cfg
        users = train["user"].to_numpy(np.int32)
        items = train["item"].to_numpy(np.int32)
        counts = np.bincount(items, minlength=self.data.n_items) + 1.0
        log_q = tf.constant(np.log(counts / counts.sum()), tf.float32)
        opt = keras.optimizers.Adam(cfg.retr_lr)
        opt.build(self.model.trainable_variables)
        tau = cfg.temperature

        @tf.function(input_signature=[tf.TensorSpec([None], tf.int32), tf.TensorSpec([None], tf.int32)])
        def step(uid, iid):
            with tf.GradientTape() as tape:
                u, v = self._user_vec(uid), self._item_vec(iid)
                logits = tf.matmul(u, v, transpose_b=True) / tau - tf.gather(log_q, iid)[None, :]
                n = tf.shape(iid)[0]
                dup = tf.cast(tf.equal(iid[:, None], iid[None, :]), tf.float32) - tf.eye(n)  # faux négatifs
                logits -= 1e9 * dup
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(tf.range(n), logits))
            grads = tape.gradient(loss, self.model.trainable_variables)
            opt.apply_gradients(zip(grads, self.model.trainable_variables))
            return loss

        rng = np.random.default_rng(cfg.seed)
        best, best_w, bad = -1.0, None, 0
        bs = min(cfg.retr_batch_size, len(users))
        for epoch in range(1, cfg.retr_epochs + 1):
            perm = rng.permutation(len(users))
            losses = [
                step(tf.constant(users[perm[s:s + bs]]), tf.constant(items[perm[s:s + bs]])).numpy()
                for s in range(0, len(perm) - bs + 1, bs)
            ]
            score = self.validate(valid_users, valid_target, valid_seen, k=50)
            if verbose:
                print(f"[retriever] epoch {epoch:02d}  loss={np.mean(losses):.4f}  valid recall@50={score:.4f}")
            if score > best:
                best, best_w, bad = score, self.model.get_weights(), 0
            else:
                bad += 1
                if bad >= cfg.retr_patience:
                    break
        self.model.set_weights(best_w)
        return best

    def validate(self, users, target, seen, k=50) -> float:
        _, ids = exact_topk(self.user_vectors(users), self.item_vectors(), seen[users], k)
        hits = hit_matrix(ids, users, target)
        n_rel = np.asarray(target[users].sum(1)).ravel().astype(int)
        return float(recall_at_k(hits, n_rel, k).mean())


# --------------------------------------------------------------------------- recherche
def exact_topk(user_vecs, item_vecs, seen: sp.csr_matrix, k: int, chunk=1024):
    """Top-k exact par produit scalaire en excluant les items déjà vus (référence / baselines)."""
    all_s, all_i = [], []
    for s in range(0, len(user_vecs), chunk):
        scores = user_vecs[s:s + chunk] @ item_vecs.T
        r, c = seen[s:s + chunk].nonzero()
        scores[r, c] = -np.inf
        top = np.argpartition(-scores, min(k, scores.shape[1] - 1), axis=1)[:, :k]
        top_s = np.take_along_axis(scores, top, 1)
        order = np.argsort(-top_s, axis=1)
        all_i.append(np.take_along_axis(top, order, 1))
        all_s.append(np.take_along_axis(top_s, order, 1))
    return np.concatenate(all_s), np.concatenate(all_i)


class FaissIndex:
    """Index de similarité (produit scalaire sur vecteurs normalisés) : flat, HNSW ou IVF."""

    def __init__(self, kind="flat", hnsw_m=32, ef_construction=200, ef_search=128, nprobe=16):
        self.kind, self.hnsw_m, self.ef_c, self.ef_s, self.nprobe = kind, hnsw_m, ef_construction, ef_search, nprobe
        self.index = None

    def build(self, vectors: np.ndarray):
        v = np.ascontiguousarray(vectors, dtype=np.float32)
        n, d = v.shape
        if faiss is None:
            self.index = _NumpyFlatIP(v)
            return self
        if self.kind == "flat":
            idx = faiss.IndexFlatIP(d)
        elif self.kind == "hnsw":
            idx = faiss.IndexHNSWFlat(d, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            idx.hnsw.efConstruction, idx.hnsw.efSearch = self.ef_c, self.ef_s
        elif self.kind == "ivf":
            nlist = max(1, int(4 * np.sqrt(n)))
            idx = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
            idx.train(v)
            idx.nprobe = self.nprobe
        else:
            raise ValueError(f"Index inconnu : {self.kind}")
        idx.add(v)
        self.index = idx
        return self

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    def search(self, queries: np.ndarray, k: int):
        return self.index.search(np.ascontiguousarray(queries, dtype=np.float32), k)

    def save(self, path: str):
        if faiss is None:
            np.save(path + ".npy", self.index.vectors)
        else:
            faiss.write_index(self.index, path)


class _NumpyFlatIP:
    """Repli exact (même interface que faiss.IndexFlatIP) quand FAISS ne peut pas être chargé."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors
        self.ntotal = len(vectors)

    def search(self, queries: np.ndarray, k: int):
        scores = queries @ self.vectors.T
        top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        top_s = np.take_along_axis(scores, top, 1)
        order = np.argsort(-top_s, axis=1)
        return np.take_along_axis(top_s, order, 1), np.take_along_axis(top, order, 1)


def generate_candidates(index: FaissIndex, user_vecs, seen_rows: sp.csr_matrix, n_candidates: int, overfetch: int):
    """Top-N voisins FAISS par utilisateur, après retrait des items déjà vus.

    On sur-récupère `overfetch` voisins pour compenser le filtrage. Rembourrage : id -1, score -inf.
    """
    k = min(index.ntotal, n_candidates + overfetch)
    scores, ids = index.search(user_vecs, k)
    n = len(user_vecs)
    out_ids = np.full((n, n_candidates), -1, dtype=np.int64)
    out_scores = np.full((n, n_candidates), -np.inf, dtype=np.float32)
    for u in range(n):
        seen = seen_rows.indices[seen_rows.indptr[u]:seen_rows.indptr[u + 1]]
        keep = (ids[u] >= 0) & ~np.isin(ids[u], seen)
        kept_ids, kept_scores = ids[u][keep][:n_candidates], scores[u][keep][:n_candidates]
        out_ids[u, :len(kept_ids)] = kept_ids
        out_scores[u, :len(kept_ids)] = kept_scores
    return out_ids, out_scores
