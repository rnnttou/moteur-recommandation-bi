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
