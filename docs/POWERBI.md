# Dashboard Power BI

## 1. Import
Power BI Desktop → *Obtenir les données* → *Dossier* → `outputs/bi/` (ou chaque CSV).

## 2. Modèle de données (étoile)

```
dim_user (user) 1──* fact_recommendations (user, item) *──1 dim_item (item)
dim_user (user) 1──* fact_user_metrics (user, model)
dim_user (user) 1──* fact_interactions (user, item)    *──1 dim_item (item)
fact_metrics, fact_diversity : indépendantes (filtrées par `model`)
```
Relations 1-à-plusieurs, filtrage unidirectionnel des dimensions vers les faits.

## 3. Mesures DAX

```DAX
Recall@10 =
CALCULATE ( AVERAGE ( fact_user_metrics[recall@10] ) )

NDCG@10 =
CALCULATE ( AVERAGE ( fact_user_metrics[ndcg@10] ) )

Gain NDCG@10 vs Retrieval seul =
VAR _full = CALCULATE ( [NDCG@10], fact_user_metrics[model] = "two_stage" )
VAR _retr = CALCULATE ( [NDCG@10], fact_user_metrics[model] = "retrieval_only" )
RETURN DIVIDE ( _full - _retr, _retr )

Taux de reco pertinentes =
DIVIDE ( SUM ( fact_recommendations[is_hit] ), COUNTROWS ( fact_recommendations ) )

Couverture catalogue =
CALCULATE ( MAX ( fact_diversity[catalog_coverage] ), fact_diversity[model] = "two_stage" )

Ecart exposition genre =
AVERAGE ( fact_genre_exposure[recommended_share_top10] ) - AVERAGE ( fact_genre_exposure[test_share] )
```

## 4. Pages proposées

1. **Vue d'ensemble** : cartes (Recall@10, NDCG@10, couverture), histogramme groupé *modèle × k* depuis `fact_metrics` (popularité vs retrieval vs two-stage), carte du plafond `candidate_ceiling`.
2. **Segments** : NDCG@10 par tranche d'âge, profession et sexe (matrice + barres) → où le moteur est-il moins performant ?
3. **Contenu & diversité** : exposition par genre (catalogue / consommé / recommandé), popularité moyenne, couverture du catalogue.
4. **Explorateur de recommandations** : segment utilisateur → top-20 (titre, rang, score, hit), avec tooltip sur le score du retriever vs. du classeur.

## 5. Lecture métier
- Un NDCG qui s'effondre sur un segment = manque de données (cold-start) à traiter spécifiquement.
- `recommended_share` ≫ `test_share` sur un genre = sur-exposition (biais de popularité à corriger par diversification).
- Gain `two_stage` vs `retrieval_only` = valeur ajoutée du classeur, à mettre en regard de son coût de calcul.

## 6. Pas-à-pas pour construire le `.pbix`

Le `.pbix` n'est pas versionné : il se construit dans Power BI Desktop (Windows) en une demi-heure à partir de ce qui suit. `docs/dashboard_page1-3.png` montrent le contenu visé (aperçus matplotlib, pas des captures Power BI).

1. **Importer** les 8 CSV de `outputs/bi/` (*Obtenir les données → Texte/CSV*, ou *Dossier*). Vérifier les types : `user`, `item`, `k`, `rank` en entier, `recall@10`… en décimal.
2. **Relations** (vue Modèle) : `dim_user[user]` → `fact_user_metrics[user]`, `fact_recommendations[user]`, `fact_interactions[user]` ; `dim_item[item]` → `fact_recommendations[item]`, `fact_interactions[item]`. Un seul sens de filtrage. Ne pas relier `fact_metrics`, `fact_diversity`, `fact_genre_exposure`.
3. **Pièges** :
   - `fact_user_metrics` contient les **3 modèles** : sans filtre sur `model`, `AVERAGE` mélange les modèles. Soit un segment « Modèle » (sélection simple, `two_stage` par défaut) sur chaque page, soit les mesures ci-dessous, qui filtrent explicitement.
   - `fact_user_metrics` reprend déjà `age_group`, `occupation`, `gender` : utilisez ces colonnes pour les axes de la page Segments (pas besoin de passer par `dim_user`).
   - Trier `age_group` manuellement (`<18, 18-24, 25-34, 35-44, 45-49, 50-55, 56+`) : l'ordre alphabétique place `<18` après `56+`. Créer une colonne de tri.
4. **Mesures** : celles de la section 3, plus celles de la section 7 ci-dessous.
5. **Pages** :
   - *Vue d'ensemble* : 4 cartes (Recall@10, NDCG@10, Couverture, `candidate_ceiling` recall@100) + histogrammes groupés `fact_metrics` (axe `k`, légende `model`, filtre `metric`).
   - *Segments* : barres `NDCG@10 two_stage` par `age_group`, par `gender`, et par `occupation` (barres triées, filtre visuel sur `Effectif segment ≥ 10`), ligne de référence = moyenne globale.
   - *Contenu & diversité* : barres groupées `catalog_share`, `test_share`, `recommended_share_top10` par `genre` (tri sur `Ecart exposition genre par genre`) + colonne de couverture par modèle.
   - *Explorateur* : segment `occupation`/`age_group` → tableau `fact_recommendations` filtré sur `rank ≤ 20`, avec `dim_item[title]`, `ranker_score`, `retrieval_score`, `is_hit`.
6. **Enregistrer** sous `docs/dashboard.pbix`, exporter 2 à 3 pages en PNG (*Fichier → Exporter*, ou capture d'écran) et remplacer les aperçus.

## 7. Mesures supplémentaires

```DAX
NDCG@10 two_stage =
CALCULATE ( AVERAGE ( fact_user_metrics[ndcg@10] ), fact_user_metrics[model] = "two_stage" )

NDCG@10 popularité =
CALCULATE ( AVERAGE ( fact_user_metrics[ndcg@10] ), fact_user_metrics[model] = "popularity" )

Gain vs popularité =
DIVIDE ( [NDCG@10 two_stage] - [NDCG@10 popularité], [NDCG@10 popularité] )

Effectif segment =
CALCULATE ( DISTINCTCOUNT ( fact_user_metrics[user] ), fact_user_metrics[model] = "two_stage" )

Hit@10 two_stage =
CALCULATE ( AVERAGE ( fact_user_metrics[hit@10] ), fact_user_metrics[model] = "two_stage" )

Ecart NDCG vs moyenne =
VAR _global = CALCULATE ( [NDCG@10 two_stage], ALL ( fact_user_metrics[age_group], fact_user_metrics[occupation], fact_user_metrics[gender] ) )
RETURN DIVIDE ( [NDCG@10 two_stage] - _global, _global )

Ratio exposition genre =
DIVIDE ( SUM ( fact_genre_exposure[recommended_share_top10] ), SUM ( fact_genre_exposure[test_share] ) )

Ecart exposition genre par genre =
SUM ( fact_genre_exposure[recommended_share_top10] ) - SUM ( fact_genre_exposure[test_share] )
```

Note : la mesure `Ecart exposition genre` de la section 3 moyenne l'écart sur tous les genres (≈ 0 par construction, les parts sommant à 1). Utiliser `Ecart exposition genre par genre` sur l'axe `genre`.
