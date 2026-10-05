"""Export du résultat sous forme de modèle en étoile (CSV) pour Power BI / Tableau / Looker Studio."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import hit_matrix


def export_bi(res: dict, out_dir: str) -> Path:
    out = Path(out_dir) / "bi"
    out.mkdir(parents=True, exist_ok=True)
    data, splits, users = res["data"], res["splits"], res["users"]

    # --- dimensions
    dim_user = data.users[["user", "gender", "age", "age_group", "occupation"]]
    dim_item = data.items[["item", "title", "year"]].copy()
    genres = data.item_genres
    dim_item["main_genre"] = np.where(genres.sum(1) > 0, np.array(data.genre_names)[genres.argmax(1)], "unknown")
    dim_item["n_genres"] = genres.sum(1).astype(int)
    dim_item["train_popularity"] = np.asarray(res["history"].sum(0)).ravel().astype(int)
    dim_user.to_csv(out / "dim_user.csv", index=False)
    dim_item.to_csv(out / "dim_item.csv", index=False)

    # --- faits : interactions étiquetées par split
    fact = pd.concat([splits.retriever.assign(split="retriever"), splits.ranker.assign(split="ranker"),
                      splits.valid.assign(split="valid"), splits.test.assign(split="test")])
    fact.to_csv(out / "fact_interactions.csv", index=False)

    # --- faits : métriques globales et par utilisateur
    res["metrics"].to_csv(out / "fact_metrics.csv", index=False)
    res["per_user"].merge(dim_user, on="user").to_csv(out / "fact_user_metrics.csv", index=False)
    res["diversity"].to_csv(out / "fact_diversity.csv", index=False)

    # --- faits : recommandations (top-20 du pipeline complet)
    o = res["output"]
    ranked = o["ranked"][:, :20]
    hits = hit_matrix(ranked, users, res["h_test"])
    n, k = ranked.shape
    recs = pd.DataFrame({
        "user": np.repeat(users, k),
        "rank": np.tile(np.arange(1, k + 1), n),
        "item": ranked.ravel(),
        "ranker_score": o["ranked_scores"][:, :20].ravel(),
        "retrieval_score": o["ranked_retrieval_scores"][:, :20].ravel(),
        "is_hit": hits.ravel().astype(int),
    })
    recs[recs["item"] >= 0].to_csv(out / "fact_recommendations.csv", index=False)

    # --- exposition par genre : catalogue vs ventes réelles (test) vs recommandations
    g_cat = genres.mean(0)
    test_items = splits.test["item"].to_numpy()
    g_test = genres[test_items].sum(0) / genres[test_items].sum()
    r10 = res["runs"]["two_stage"][:, :10]
    g_rec = genres[r10[r10 >= 0]].sum(0) / genres[r10[r10 >= 0]].sum()
    pd.DataFrame({
        "genre": data.genre_names,
        "catalog_share": g_cat / g_cat.sum(),
        "test_share": g_test,
        "recommended_share_top10": g_rec,
    }).to_csv(out / "fact_genre_exposure.csv", index=False)
    return out


def plot_metrics(metrics: pd.DataFrame, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = ["popularity", "retrieval_only", "two_stage"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ["recall", "ndcg"]):
        df = metrics[(metrics["metric"] == metric) & metrics["model"].isin(models)]
        for m in models:
            d = df[df["model"] == m].sort_values("k")
            ax.plot(d["k"], d["value"], marker="o", label=m)
        ax.set_title(f"{metric}@k")
        ax.set_xlabel("k")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
