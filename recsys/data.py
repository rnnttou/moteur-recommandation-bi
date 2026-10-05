from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
GENRES_ML100K = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
AGE_BINS = [0, 18, 25, 35, 45, 50, 56, 200]
AGE_LABELS = ["<18", "18-24", "25-34", "35-44", "45-49", "50-55", "56+"]


@dataclass
class Dataset:
    name: str
    interactions: pd.DataFrame  # user, item, rating, timestamp (ids encodés 0..n-1)
    users: pd.DataFrame  # une ligne par user (index = id)
    items: pd.DataFrame  # une ligne par item (index = id) + colonnes de genres 0/1
    genre_names: list

    @property
    def n_users(self) -> int:
        return len(self.users)

    @property
    def n_items(self) -> int:
        return len(self.items)

    @property
    def item_genres(self) -> np.ndarray:
        return self.items[self.genre_names].to_numpy(np.float32)


@dataclass
class Splits:
    retriever: pd.DataFrame  # historique ancien : entraîne l'étape 1
    ranker: pd.DataFrame  # entraîne l'étape 2 (labels)
    valid: pd.DataFrame  # sélection de modèles / early stopping
    test: pd.DataFrame  # évaluation finale


# --------------------------------------------------------------------------- chargement
def _finalize_users(users: pd.DataFrame) -> pd.DataFrame:
    users = users.sort_values("user").reset_index(drop=True)
    users["age_group"] = pd.cut(users["age"], AGE_BINS, labels=AGE_LABELS, right=False).astype(str)
    users["gender_id"] = (users["gender"] == "F").astype(int)
    users["age_id"] = pd.Categorical(users["age_group"], categories=AGE_LABELS).codes.astype(int)
    users["occupation_id"] = pd.factorize(users["occupation"], sort=True)[0].astype(int)
    return users


def load_movielens_100k(data_dir: str) -> Dataset:
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    folder = root / "ml-100k"
    if not folder.exists():
        archive = root / "ml-100k.zip"
        print(f"Téléchargement de MovieLens 100K -> {archive}")
        urllib.request.urlretrieve(ML100K_URL, archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(root)

    inter = pd.read_csv(folder / "u.data", sep="\t", names=["user", "item", "rating", "timestamp"])
    inter[["user", "item"]] -= 1

    users = pd.read_csv(folder / "u.user", sep="|", names=["user", "age", "gender", "occupation", "zip"])
    users["user"] -= 1

    cols = ["item", "title", "release_date", "video_release", "url"] + GENRES_ML100K
    items = pd.read_csv(folder / "u.item", sep="|", names=cols, encoding="latin-1")
    items["item"] -= 1
    year = pd.to_datetime(items["release_date"], format="%d-%b-%Y", errors="coerce").dt.year
    items["year"] = year.fillna(year.median()).astype(int)
    items = items.sort_values("item").reset_index(drop=True)
    items = items[["item", "title", "year"] + GENRES_ML100K]

    return Dataset("ml-100k", inter, _finalize_users(users), items, list(GENRES_ML100K))


def make_synthetic(n_users=2000, n_items=1500, n_genres=18, mean_inter=60, seed=0) -> Dataset:
    """Jeu de données hors-ligne avec préférences latentes par genre + popularité."""
    rng = np.random.default_rng(seed)
    genres = [f"Genre_{i:02d}" for i in range(n_genres)]
    item_genre = (rng.random((n_items, n_genres)) < 0.12).astype(np.float32)
    item_genre[item_genre.sum(1) == 0, rng.integers(0, n_genres)] = 1.0
    user_pref = rng.dirichlet(np.full(n_genres, 0.3), n_users)
    log_pop = rng.normal(0, 1, n_items)
    affinity = user_pref @ item_genre.T
    affinity = (affinity - affinity.mean()) / affinity.std()

    rows = []
    for u in range(n_users):
        n_u = int(np.clip(rng.poisson(mean_inter), 15, n_items // 2))
        util = 1.5 * affinity[u] + 0.7 * log_pop + rng.gumbel(size=n_items)
        chosen = np.argpartition(-util, n_u)[:n_u]
        rating = np.clip(np.rint(3.3 + 0.9 * affinity[u, chosen] + rng.normal(0, 0.8, n_u)), 1, 5)
        ts = rng.permutation(n_u)
        rows.append(pd.DataFrame({"user": u, "item": chosen, "rating": rating, "timestamp": ts}))
    inter = pd.concat(rows, ignore_index=True)

    users = pd.DataFrame({
        "user": np.arange(n_users),
        "age": rng.integers(15, 70, n_users),
        "gender": rng.choice(["M", "F"], n_users),
        "occupation": rng.choice(["student", "engineer", "artist", "teacher", "other"], n_users),
    })
    items = pd.DataFrame({
        "item": np.arange(n_items),
        "title": [f"Item {i}" for i in range(n_items)],
        "year": rng.integers(1980, 2024, n_items),
    })
    items[genres] = item_genre.astype(int)
    return Dataset("synthetic", inter, _finalize_users(users), items, genres)


def load_dataset(name: str, data_dir: str, seed: int = 0) -> Dataset:
    if name == "ml-100k":
        return load_movielens_100k(data_dir)
    if name == "synthetic":
        return make_synthetic(seed=seed)
    raise ValueError(f"Dataset inconnu : {name}")


# --------------------------------------------------------------------------- découpage
def temporal_split(
    interactions: pd.DataFrame,
    positive_threshold: float = 4.0,
    min_positives: int = 5,
    test_frac: float = 0.2,
    valid_frac: float = 0.1,
    ranker_frac: float = 0.2,
) -> Splits:
    """Découpage chronologique par utilisateur : retriever | ranker | valid | test.

    Le passé le plus ancien entraîne le retriever, la tranche suivante fournit les labels du
    classeur (qui n'a donc jamais vu ces positifs côté retriever : pas de fuite), puis
    validation et test ne contiennent que des interactions plus récentes.
    """
    pos = interactions[interactions["rating"] >= positive_threshold]
    pos = pos.sort_values(["user", "timestamp", "item"]).reset_index(drop=True)
    n = pos.groupby("user")["item"].transform("size").to_numpy()
    keep = n >= min_positives
    pos, n = pos[keep].reset_index(drop=True), n[keep]

    from_end = n - 1 - pos.groupby("user").cumcount().to_numpy()
    n_test = np.maximum(1, np.rint(test_frac * n))
    n_valid = np.maximum(1, np.rint(valid_frac * n))
    n_rank = np.maximum(1, np.rint(ranker_frac * n))
    part = np.select(
        [from_end < n_test, from_end < n_test + n_valid, from_end < n_test + n_valid + n_rank],
        ["test", "valid", "ranker"],
        default="retriever",
    )
    return Splits(**{k: pos[part == k].reset_index(drop=True) for k in ("retriever", "ranker", "valid", "test")})


def to_csr(df: pd.DataFrame, n_users: int, n_items: int) -> sp.csr_matrix:
    m = sp.csr_matrix(
        (np.ones(len(df), np.float32), (df["user"].to_numpy(), df["item"].to_numpy())),
        shape=(n_users, n_items),
    )
    return binarize(m)


def binarize(m: sp.spmatrix) -> sp.csr_matrix:
    m = sp.csr_matrix(m)
    m.data = np.ones_like(m.data, dtype=np.float32)
    return m
