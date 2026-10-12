"""Rend 3 pages d'aperçu du dashboard à partir de outputs/bi/*.csv.

Les calculs reproduisent les mesures DAX de docs/POWERBI.md. Ce n'est PAS une
capture Power BI : c'est un rendu matplotlib des mêmes chiffres, utile pour
vérifier le contenu des pages avant de les construire dans Power BI Desktop.
"""
import csv
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BI, OUT = ROOT / "outputs" / "bi", ROOT / "docs"
MODELS = [("popularity", "Popularité"), ("retrieval_only", "Retrieval seul"), ("two_stage", "Pipeline complet")]
COL = {"popularity": "#9aa5b1", "retrieval_only": "#5b8def", "two_stage": "#e8590c"}


def read(name):
    with open(BI / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


um, metrics, div, genres = read("fact_user_metrics.csv"), read("fact_metrics.csv"), read("fact_diversity.csv"), read("fact_genre_exposure.csv")


def ndcg_by(col, model, order=None):
    groups = {}
    for r in um:
        if r["model"] == model:
            groups.setdefault(r[col], []).append(float(r["ndcg@10"]))
    keys = order or sorted(groups)
    return keys, [st.mean(groups[k]) for k in keys], [len(groups[k]) for k in keys]


def mval(model, metric, k):
    return next(float(r["value"]) for r in metrics if r["model"] == model and r["metric"] == metric and r["k"] == str(k))


def card(ax, title, value, sub=""):
    ax.axis("off")
    ax.text(0.5, 0.68, value, ha="center", fontsize=24, fontweight="bold", color="#1b2a41")
    ax.text(0.5, 0.35, title, ha="center", fontsize=10, color="#444")
    ax.text(0.5, 0.12, sub, ha="center", fontsize=8, color="#888")
    ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=False, ec="#d0d7de", transform=ax.transAxes))


def page1():
    fig = plt.figure(figsize=(13, 7.2))
    fig.suptitle("Page 1 - Vue d'ensemble", fontsize=15, fontweight="bold", x=0.02, ha="left")
    ndcg = mval("two_stage", "ndcg", 10)
    gain = ndcg / mval("retrieval_only", "ndcg", 10) - 1
    cov = next(float(r["catalog_coverage"]) for r in div if r["model"] == "two_stage")
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 3.2], hspace=0.35, wspace=0.15, top=0.88)
    card(fig.add_subplot(gs[0, 0]), "Recall@10", f"{mval('two_stage', 'recall', 10):.3f}")
    card(fig.add_subplot(gs[0, 1]), "NDCG@10", f"{ndcg:.3f}", f"gain vs retrieval : {gain:+.0%}")
    card(fig.add_subplot(gs[0, 2]), "Couverture catalogue", f"{cov:.0%}")
    card(fig.add_subplot(gs[0, 3]), "Plafond candidats (recall@100)", f"{mval('candidate_ceiling', 'recall', 100):.2f}")
    ks = [10, 20, 50]
    for i, (metric, lab) in enumerate([("ndcg", "NDCG@k"), ("recall", "Recall@k")]):
        ax = fig.add_subplot(gs[1, i * 2:i * 2 + 2])
        w = 0.26
        for j, (m, name) in enumerate(MODELS):
            vals = [mval(m, metric, k) for k in ks]
            ax.bar([x + (j - 1) * w for x in range(3)], vals, w, label=name, color=COL[m])
        ax.set_xticks(range(3), [f"k={k}" for k in ks])
        ax.set_title(lab, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        if i == 0:
            ax.legend(frameon=False)
    return fig


def page2():
    fig, axes = plt.subplots(1, 3, figsize=(13, 7.2), gridspec_kw={"width_ratios": [1.7, 0.7, 1.5]})
    fig.suptitle("Page 2 - Segments : NDCG@10 par profil", fontsize=15, fontweight="bold", x=0.02, ha="left")
    ages = ["<18", "18-24", "25-34", "35-44", "45-49", "50-55", "56+"]
    for ax, col, order, title in [(axes[0], "age_group", ages, "Tranche d'âge"), (axes[1], "gender", ["F", "M"], "Sexe")]:
        for j, m in enumerate(["popularity", "two_stage"]):
            keys, vals, ns = ndcg_by(col, m, order)
            ax.bar([x + (j - 0.5) * 0.38 for x in range(len(keys))], vals, 0.38, color=COL[m], label=dict(MODELS)[m])
        _, _, ns = ndcg_by(col, "two_stage", order)
        ax.set_xticks(range(len(order)), [f"{k}\n(n={n})" for k, n in zip(order, ns)], fontsize=7)
        ax.set_title(title, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    keys, vals, ns = ndcg_by("occupation", "two_stage")
    rows = sorted((v, k, n) for k, v, n in zip(keys, vals, ns) if n >= 10)
    ax = axes[2]
    ax.barh([f"{k} (n={n})" for v, k, n in rows], [v for v, k, n in rows], color=COL["two_stage"])
    ax.axvline(mval("two_stage", "ndcg", 10), color="#333", ls="--", lw=1)
    ax.set_title("Profession (n >= 10), tirets = moyenne", loc="left")
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def page3():
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.2), gridspec_kw={"width_ratios": [2.2, 1]})
    fig.suptitle("Page 3 - Contenu & diversité", fontsize=15, fontweight="bold", x=0.02, ha="left")
    g = sorted((float(r["recommended_share_top10"]) - float(r["test_share"]), r["genre"], r) for r in genres if r["genre"] != "unknown")
    ax = axes[0]
    y = range(len(g))
    ax.barh([x - 0.27 for x in y], [float(r["catalog_share"]) for _, _, r in g], 0.27, color="#c9d1d9", label="Catalogue")
    ax.barh(list(y), [float(r["test_share"]) for _, _, r in g], 0.27, color=COL["retrieval_only"], label="Consommé (test)")
    ax.barh([x + 0.27 for x in y], [float(r["recommended_share_top10"]) for _, _, r in g], 0.27, color=COL["two_stage"], label="Recommandé (top-10)")
    ax.set_yticks(list(y), [f"{n}  ({d:+.1%})" for d, n, _ in g], fontsize=8)
    ax.set_title("Part par genre, trié par écart recommandé - consommé", loc="left")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    ax.bar([n for _, n in MODELS], [float(next(r for r in div if r["model"] == m)["catalog_coverage"]) for m, _ in MODELS], color=[COL[m] for m, _ in MODELS])
    ax.set_title("Couverture du catalogue (k=10)", loc="left")
    ax.tick_params(axis="x", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


if __name__ == "__main__":
    for i, fn in enumerate([page1, page2, page3], 1):
        fig = fn()
        path = OUT / f"dashboard_page{i}.png"
        fig.savefig(path, dpi=110)
        print("écrit", path)
