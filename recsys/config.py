from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # --- données ---
    dataset: str = "ml-100k"  # "ml-100k" (téléchargé) ou "synthetic" (hors-ligne)
    data_dir: str = "data"
    output_dir: str = "outputs"
    seed: int = 42
    positive_threshold: float = 4.0  # note >= seuil => interaction positive (feedback implicite)
    min_positives: int = 5  # utilisateurs avec moins de positifs exclus du protocole

    # --- étape 1 : génération de candidats (two-tower + FAISS) ---
    embedding_dim: int = 64
    hidden_dim: int = 128
    temperature: float = 0.1
    retr_epochs: int = 40
    retr_batch_size: int = 512
    retr_lr: float = 1e-3
    retr_patience: int = 5
    index_type: str = "flat"  # "flat" (exact) | "hnsw" | "ivf"
    n_candidates: int = 100
    overfetch: int = 300  # voisins supplémentaires récupérés pour filtrer les items déjà vus

    # --- étape 2 : modèle de classement ---
    ranker_embedding_dim: int = 16
    ranker_hidden: tuple = (128, 64)
    ranker_dropout: float = 0.2
    ranker_epochs: int = 30
    ranker_batch_size: int = 1024
    ranker_lr: float = 1e-3
    ranker_patience: int = 3
    ranker_pos_weight: float = 3.0

    # --- évaluation ---
    ks: tuple = (10, 20, 50)
