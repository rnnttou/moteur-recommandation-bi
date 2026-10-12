# Moteur de recommandation personnalisée en deux étapes

**Python · TensorFlow · FAISS** — génération de candidats par embeddings, puis modèle de classement, évalués avec **NDCG@k** et **recall@k**. Les résultats sont exportés en modèle en étoile pour **Power BI / Tableau**.

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rnnttou/moteur-recommandation-bi/blob/main/notebooks/demo_moteur.ipynb)

> Statut : terminé. Pipeline complet, tests unitaires et résultats ci-dessous (voir « Résultats »).

---

## 1. Problème et approche

**Le besoin métier.** Une plateforme de streaming veut augmenter l'engagement : que chaque abonné trouve vite un film qu'il va réellement regarder. La solution de facilité est de recommander les films les plus populaires. Elle marche à peu près (recall@10 de 0,059 ici) mais elle sert tout le monde pareil, laisse 97 % du catalogue invisible (couverture de 3 %) et n'apprend rien du profil de chacun. L'objectif est donc de **personnaliser sans s'enfermer dans les blockbusters**, et de **mesurer** à qui le moteur profite et ce qu'il met en avant.

**La contrainte technique.** Un catalogue de milliers (ou millions) d'items ne peut pas être scoré en entier par un modèle riche. Le standard industriel est une cascade : une étape rapide qui ne rate presque rien, puis une étape précise qui ordonne.

```
                 tout le catalogue (N items)
                           │
        ┌──────────────────▼──────────────────┐
        │ ÉTAPE 1 — Génération de candidats   │  rapide, rappel élevé
        │ Two-tower (TensorFlow) + index FAISS│  → top 100 par utilisateur
        └──────────────────┬──────────────────┘
                           │ 100 candidats
        ┌──────────────────▼──────────────────┐
        │ ÉTAPE 2 — Classement                │  précis, plus coûteux
        │ MLP + features croisées (TensorFlow)│  → top 10 / 20 / 50
        └──────────────────┬──────────────────┘
                           │
              Évaluation : recall@k, NDCG@k
                           │
              Export CSV (schéma en étoile) → Power BI
```

| Étape | Rôle | Métrique clé |
|---|---|---|
| 1. Retrieval | ne rien rater : récupérer les bons items parmi des milliers | **recall@100** (plafond du système) |
| 2. Ranking | bien ordonner les candidats | **NDCG@k**, recall@k |

## 2. Données

- **MovieLens 100K** (943 utilisateurs, 1 682 films, 100 000 notes), téléchargé automatiquement depuis GroupLens.
- Un mode `--dataset synthetic` génère un jeu hors-ligne (préférences latentes par genre + popularité).
- **Feedback implicite** : note ≥ 4 = interaction positive.
- Features : utilisateur (sexe, tranche d'âge, profession), item (genres, année, popularité).

## 3. Protocole d'évaluation (sans fuite)

Découpage **chronologique par utilisateur** (≥ 5 positifs) :

```
 passé ─────────────────────────────────────────────► récent
 [ retriever 50% ][ ranker 20% ][ valid 10% ][ test 20% ]
```

- `retriever` entraîne l'étape 1 ;
- `ranker` fournit les **labels** de l'étape 2 : le retriever ne les a jamais vus, donc ses scores sont réalistes (pas de surapprentissage transmis au classeur) ;
- `valid` : early stopping des deux modèles ;
- `test` : évaluation finale, historique = tout ce qui précède. Les items déjà vus sont exclus des recommandations.
- La popularité utilisée comme feature est calculée sur le seul segment `retriever` pour ne pas fuiter les labels.

### Métriques (binaires, par utilisateur puis moyennées)

- **recall@k** = |recommandés@k ∩ pertinents| / |pertinents|
- **NDCG@k** = DCG@k / IDCG@k, avec DCG = Σ rel_j / log₂(j+1) : récompense de placer les bons items **haut** dans la liste
- **hit-rate@k**, **couverture du catalogue**, **popularité moyenne** des recommandations (angle métier / diversité)

Baselines comparées : **popularité**, **retrieval seul**, **pipeline complet (retrieval + ranking)**, plus le **plafond de candidats** (recall@100).

## Résultats

MovieLens 100K, jeu de **test** chronologique, index FAISS HNSW, items déjà vus exclus (`python run_pipeline.py --index hnsw`) :

| Modèle | recall@10 | NDCG@10 | recall@20 | NDCG@20 | NDCG@50 |
|---|---|---|---|---|---|
| Popularité | 0,059 | 0,077 | 0,107 | 0,090 | 0,129 |
| Retrieval seul (two-tower + FAISS) | 0,077 | 0,079 | 0,141 | 0,101 | 0,146 |
| **Pipeline complet (retrieval + ranking)** | **0,094** | **0,109** | **0,168** | **0,129** | **0,173** |

- Le classeur apporte **+38 % de NDCG@10** par rapport au retrieval seul (0,079 → 0,109) et **+42 %** par rapport à la popularité.
- Plafond de l'étape 1 : **recall@100 = 0,42** (hit-rate@100 = 0,91), donc borne supérieure du recall du classeur.
- Diversité (k = 10) : couverture du catalogue de 22 % pour le pipeline complet, contre 3 % pour la popularité (le retrieval seul atteint 40 %, au prix d'un meilleur classement perdu).

![Métriques](docs/metrics.png)

Résultats d'une exécution unique (une graine) sur un petit jeu : les écarts relatifs sont fiables, les valeurs absolues restent indicatives.

## 4. Détail des modèles

### Étape 1 — Two-tower + FAISS (`recsys/retrieval.py`)
- **Tour utilisateur** : embedding de l'id + sexe + âge + profession → MLP → vecteur L2-normalisé.
- **Tour item** : embedding de l'id + genres multi-hot → MLP → vecteur L2-normalisé.
- **Perte** : softmax avec **négatifs in-batch**, température τ = 0,1, **correction logQ** (compense le biais de popularité de l'échantillonnage) et masquage des faux négatifs (même item répété dans le batch).
- **Sélection** : early stopping sur le recall@50 de validation.
- **Index FAISS** sur les vecteurs d'items (produit scalaire = cosinus) :
  - `flat` : exact (référence) ;
  - `hnsw` : graphe approché, latence quasi constante ;
  - `ivf` : partitionnement approché.
  `scripts/benchmark_faiss.py` mesure le compromis recall / latence sur de gros catalogues.
- Les items déjà vus sont retirés après recherche (sur-récupération `overfetch`).

### Étape 2 — Classement (`recsys/ranking.py`)
Modèle pointwise (MLP, sortie sigmoïde) sur chaque paire (utilisateur, candidat) :

| Famille | Features |
|---|---|
| Retrieval | score cosinus, écart au meilleur candidat, log du rang |
| Item | log-popularité, année, genres |
| Croisées | affinité du profil de genres de l'historique avec l'item, écart de popularité vs. l'historique |
| Utilisateur | taille de l'historique, embedding id ⊙ embedding item |

Perte : entropie croisée pondérée (positifs rares), early stopping sur l'AUC de validation.

## 5. Volet BI

`export_bi` écrit dans `outputs/bi/` un **modèle en étoile** :

| Table | Type | Contenu |
|---|---|---|
| `dim_user.csv` | dimension | sexe, âge, tranche d'âge, profession |
| `dim_item.csv` | dimension | titre, année, genre principal, popularité |
| `fact_interactions.csv` | fait | interactions étiquetées par split |
| `fact_recommendations.csv` | fait | top-20 par utilisateur : rang, scores, hit |
| `fact_metrics.csv` | fait | recall / NDCG / hit par modèle et par k |
| `fact_user_metrics.csv` | fait | métriques par utilisateur + attributs → analyse par segment |
| `fact_diversity.csv` | fait | couverture du catalogue, popularité moyenne |
| `fact_genre_exposure.csv` | fait | part de chaque genre : catalogue vs. consommé vs. recommandé |

### Traçabilité : du pipeline aux tables BI

| Table | Produite à partir de (code) | Grain (1 ligne = …) | Question métier |
|---|---|---|---|
| `dim_user` | `data.users` ([data.py](recsys/data.py)) | un utilisateur | segmenter par âge / profession / sexe |
| `dim_item` | `data.items` + popularité de l'historique avant test | un film | titre, genre principal, popularité |
| `fact_interactions` | `splits` ([data.py](recsys/data.py)), positifs uniquement (note ≥ 4) | une interaction positive, étiquetée `retriever` / `ranker` / `valid` / `test` | vérifier le découpage chronologique |
| `fact_recommendations` | sortie du classeur (`output["ranked"]`), pipeline complet seulement | un (utilisateur, rang), top-20 ; `is_hit` = item présent dans le test | explorer les recommandations et leurs scores |
| `fact_metrics` | `summarize(...)` ([metrics.py](recsys/metrics.py)) | un (modèle, métrique, k), moyenne sur les utilisateurs de test | comparer popularité / retrieval / pipeline complet |
| `fact_user_metrics` | `evaluate_ranking(...)` joint à `dim_user` | un (utilisateur, modèle), les 3 modèles | où le moteur est-il moins performant ? |
| `fact_diversity` | `catalog_coverage`, `mean_log_popularity` (k = 10) | un modèle | biais de popularité, couverture |
| `fact_genre_exposure` | genres des items : catalogue, test, top-10 du pipeline complet | un genre | sur-exposition de certains genres |

Notes de lecture :
- Les grains diffèrent : `fact_recommendations` ne contient que le pipeline complet, alors que `fact_user_metrics` contient les trois modèles. Filtrer sur `model` pour comparer.
- `fact_user_metrics` reprend les attributs de `dim_user` (dénormalisation volontaire, pour filtrer sans relation supplémentaire).
- Les chiffres de la section « Résultats » proviennent de `fact_metrics.csv` / `outputs/metrics.json` : ce sont les mêmes valeurs que celles affichées dans le dashboard.
- **Le fichier `.pbix` n'est pas dans le repo** : il doit être construit dans Power BI Desktop (pas disponible dans l'environnement où ce dépôt a été préparé). [docs/POWERBI.md](docs/POWERBI.md) donne le modèle, les mesures DAX et le pas-à-pas des 3 pages. Les images `docs/dashboard_page*.png` sont des **aperçus matplotlib** des mêmes calculs (`scripts/build_dashboard_preview.py`), pas des captures Power BI.

Pages de dashboard et mesures DAX : voir [docs/POWERBI.md](docs/POWERBI.md).

Questions métier auxquelles le dashboard répond : le moteur sert-il mieux certains segments (âge, profession) ? Sur-expose-t-il certains genres ? Quel est le gain réel du classeur sur le retrieval seul ? Réponses ci-dessous.

## Conclusions métier

Tirées des tables `outputs/bi/` (pipeline complet, jeu de test, 938 utilisateurs), pages Segments et Contenu du dashboard.

![Vue d'ensemble](docs/dashboard_page1.png)
![Segments](docs/dashboard_page2.png)
![Contenu et diversité](docs/dashboard_page3.png)

*Aperçus matplotlib des mesures du dashboard, pas des captures Power BI.*

### Quels segments sont mal servis ?

Moyenne globale : NDCG@10 = 0,109, hit@10 = 0,50.

| Segment | n | NDCG@10 | Lecture |
|---|---|---|---|
| Profession *writer* | 45 | **0,060** | Le plus faible, 45 % sous la moyenne ; seul écart net sur un effectif correct |
| Moins de 18 ans | 36 | 0,089 | NDCG bas et hit@10 de 0,39 ; en revanche **x3,5 mieux que la popularité** (0,025), donc la personnalisation aide beaucoup |
| 18-24 ans | 198 | 0,104 | **Quasi pas de gain face à la popularité** (0,100) et hit@10 de 0,44 : le plus gros segment est celui où le moteur apporte le moins |
| 45-49 ans | 79 | 0,097 | Sous la moyenne, hit@10 de 0,43 |
| Femmes | 271 | 0,103 | Léger écart vs hommes (0,112), à surveiller, pas alarmant |

Les mieux servis : *executive* (0,168), *engineer* (0,147), 25-34 ans (0,119).

Précautions : les écarts par âge et par sexe sont **modestes** (quelques millièmes de NDCG), sans intervalle de confiance, sur une seule graine. Les professions à faible effectif (*doctor* n=7, *homemaker* n=7, *salesman* n=12) ne sont pas interprétables. Le NDCG croît aussi mécaniquement avec le nombre d'items de test d'un utilisateur (0,08 pour moins de 5 items, 0,17 pour 20-50), donc une partie de l'écart entre segments reflète la taille de leur historique de test plutôt qu'une vraie faiblesse du modèle. Ce n'est pas une démonstration de cold-start : le jeu ne contient que des utilisateurs avec au moins 5 positifs.

### Le moteur sur-expose-t-il certains genres ?

Oui, mais **ce n'est pas un biais de popularité pur** : la popularité moyenne des recommandations est plus basse que celle de la baseline (log-popularité 4,98 contre 5,66) et la couverture est 7 fois meilleure (22 % contre 3 %). Le biais est un biais **de genre** : le moteur déplace l'exposition des genres « de masse » vers l'action.

| | Genre | Part consommée (test) | Part recommandée | Ratio |
|---|---|---|---|---|
| Sur-exposés | War | 3,7 % | 5,7 % | x1,55 |
| | Sci-Fi | 5,8 % | 8,6 % | x1,50 |
| | Adventure | 6,4 % | 8,8 % | x1,37 |
| | Action | 11,5 % | 14,9 % | x1,30 |
| Sous-exposés | Drama | 19,7 % | 12,5 % | x0,63 |
| | Comedy | 14,5 % | 9,3 % | x0,64 |
| | Documentary, Film-Noir | 0,4 % / 0,9 % | 0,1 % / 0,4 % | x0,39 / x0,41 |

Drama et Comedy sont les deux premiers genres du catalogue et de la consommation, mais perdent un tiers de leur place dans le top-10. Cause plausible, non testée : ces genres sont très hétérogènes, donc moins bien capturés par les embeddings que l'action.

### Ce que je recommanderais

1. **Corriger l'exposition par genre** : ajouter une étape de reclassement diversifiant (MMR ou quotas de genre autour de la part consommée), puis vérifier dans la page Contenu que l'écart recommandé - consommé passe sous ±2 points. Le coût attendu en NDCG est à mesurer, la couverture de 22 % laisse de la marge.
2. **Traiter les 18-24 ans en priorité** : segment le plus nombreux (21 % des utilisateurs) où le classeur n'ajoute presque rien à la popularité. Pistes : features de session récente (cf. section 8), négatifs difficiles.
3. **Ne pas sur-réagir aux petits segments** (*writer* mis à part, qui mérite une analyse des genres consommés) : collecter plus de données avant d'agir.
4. **Valider par A/B test** avant tout déploiement : tout ce qui précède est hors-ligne, sur MovieLens 100K, avec une seule graine.

## 6. Exécution

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate sous Linux/macOS)
pip install -r requirements.txt

pytest -q                          # tests unitaires (métriques, découpage)
python run_pipeline.py             # MovieLens 100K, index exact
python run_pipeline.py --index hnsw --save-artifacts
python run_pipeline.py --dataset synthetic            # hors-ligne
python scripts/benchmark_faiss.py --n-items 200000
```

Sorties : `outputs/metrics.json`, `outputs/metrics.png`, `outputs/bi/*.csv`.

> **Windows** : si l'import de `faiss` échoue (stratégie de contrôle d'application / Smart App Control bloquant une DLL non signée), le code bascule automatiquement sur un index NumPy exact (même interface, résultats identiques en `flat`). Pour tester HNSW/IVF, exécutez le projet sous WSL, Linux ou Google Colab.

## 7. Structure

```
recsys/
  config.py      hyperparamètres
  data.py        chargement, jeu synthétique, découpage chronologique
  metrics.py     recall@k, NDCG@k, hit-rate, couverture
  retrieval.py   two-tower TensorFlow, index FAISS, génération de candidats
  ranking.py     features, modèle de classement, reclassement
  pipeline.py    TwoStageRecommender + évaluation de bout en bout
  bi_export.py   export en étoile pour Power BI + graphique
run_pipeline.py  point d'entrée CLI
scripts/benchmark_faiss.py
scripts/build_dashboard_preview.py   aperçus PNG du dashboard
tests/
docs/POWERBI.md
```

## 8. Limites et pistes

- Classeur **pointwise** : passer à une perte **listwise / LambdaRank** (`tensorflow_ranking`) optimiserait directement le NDCG.
- Le retriever n'utilise pas la séquence récente de l'utilisateur : ajouter un encodeur de session (GRU / attention) à la tour utilisateur.
- MovieLens 100K est petit : l'intérêt de FAISS ANN n'apparaît qu'à grande échelle (benchmark fourni).
- Évaluation hors-ligne seulement : en production, valider par **A/B test** (CTR, conversion).
- Pistes : négatifs difficiles, calibration, diversification (MMR), suivi de dérive.
