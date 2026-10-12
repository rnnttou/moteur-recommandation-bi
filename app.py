"""Démo web (Gradio) : entraîne le moteur au démarrage, puis recommande pour un utilisateur choisi.

    python app.py            # local, http://127.0.0.1:7860
Déployé sur Hugging Face Spaces (voir deploy/DEPLOY.md).
"""
from __future__ import annotations

import gradio as gr
import pandas as pd

from recsys.config import Config
from recsys.pipeline import run

print("Entraînement du moteur (MovieLens 100K)...")
RES = run(Config(dataset="ml-100k", index_type="hnsw"))
DATA, OUT, USERS = RES["data"], RES["output"], RES["users"]
POS = {int(u): i for i, u in enumerate(USERS)}
ITEMS, GENRES = DATA.items, DATA.genre_names
LIKED = DATA.interactions[DATA.interactions.rating >= 4].sort_values("timestamp")
TEST = RES["splits"].test


def genres(i):
    return ", ".join(g for g in GENRES if ITEMS.loc[i, g] == 1) or "?"


def recommend(user):
    user = int(user)
    u = DATA.users.loc[user]
    profile = f"**Utilisateur {user}** : {u.gender}, {u.age} ans, {u.occupation}"
    seen = [{"Film": ITEMS.loc[i, "title"], "Genres": genres(i)} for i in LIKED[LIKED.user == user].item.tail(8)]
    test_items = set(TEST[TEST.user == user].item)
    rows = []
    for r, (i, s) in enumerate(zip(OUT["ranked"][POS[user]][:10], OUT["ranked_scores"][POS[user]][:10]), 1):
        rows.append({"Rang": r, "Film": ITEMS.loc[i, "title"], "Genres": genres(i), "Score": round(float(s), 3),
                     "Aimé ensuite ?": "oui" if i in test_items else ""})
    return profile, pd.DataFrame(seen), pd.DataFrame(rows)


def metrics_tables():
    m = RES["metrics"]
    m = m[m.metric.isin(["recall", "ndcg"]) & m.k.isin([10, 20, 50])]
    t = m.pivot_table(index="model", columns=["metric", "k"], values="value").round(3)
    t.columns = [f"{a}@{b}" for a, b in t.columns]
    t = t.reset_index().replace({"popularity": "Popularité", "retrieval_only": "Retrieval seul", "two_stage": "Pipeline complet"})
    d = RES["diversity"].round(3)
    d["model"] = d["model"].replace({"popularity": "Popularité", "retrieval_only": "Retrieval seul", "two_stage": "Pipeline complet"})
    return t, d


with gr.Blocks(title="Moteur de recommandation en deux étapes") as demo:
    gr.Markdown(
        "# Moteur de recommandation en deux étapes\n"
        "Two-tower (TensorFlow) + index FAISS pour générer 100 candidats, puis classeur pour ordonner. "
        "Données : MovieLens 100K. Les films déjà vus sont exclus."
    )
    with gr.Tab("Recommandations"):
        user = gr.Dropdown([int(u) for u in USERS], value=int(USERS[0]), label="Utilisateur")
        profile = gr.Markdown()
        gr.Markdown("**Derniers films aimés (note ≥ 4)**")
        seen = gr.Dataframe(interactive=False)
        gr.Markdown("**Top-10 recommandé**")
        recs = gr.Dataframe(interactive=False)
        user.change(recommend, user, [profile, seen, recs])
        demo.load(recommend, user, [profile, seen, recs])
    with gr.Tab("Métriques (jeu de test)"):
        t, d = metrics_tables()
        gr.Dataframe(t, interactive=False, label="recall@k et NDCG@k")
        gr.Dataframe(d, interactive=False, label="Diversité (k = 10)")
        gr.Markdown("Résultat d'une exécution unique : les écarts relatifs sont fiables, les valeurs absolues indicatives.")

if __name__ == "__main__":
    demo.launch()
