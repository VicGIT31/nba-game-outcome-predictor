# NBA Game Outcome Predictor

Prédire **quelle équipe gagne un match NBA** à partir d'environ **10 saisons de données réelles**
récupérées via le package [`nba_api`](https://github.com/swar/nba_api).

Le projet va de bout en bout : ingestion des données brutes (avec cache et rate-limiting),
construction de features ancrées dans le temps, entraînement d'une **régression logistique**,
et comparaison à une **baseline naïve** ("l'équipe à domicile gagne toujours").

---

## Motivation

L'issue d'un match NBA est en partie prévisible : l'avantage du terrain, la fatigue
(back-to-back, jours de repos), la forme récente et la qualité défensive d'une équipe
ont un effet mesurable. L'objectif est de quantifier cet effet avec un modèle **simple,
interprétable et honnête** :

- **Simple / interprétable** → régression logistique : les coefficients se lisent directement
  comme l'influence (signe + amplitude) de chaque facteur sur la probabilité de victoire.
- **Honnête** → un **split temporel** (on entraîne sur le passé, on teste sur le futur) et
  des features **décalées dans le temps** (aucune information du match courant ni du futur
  ne fuite dans les features), pour ne pas surestimer la performance.

La barre à battre n'est pas 50 % mais la **baseline "domicile gagne toujours"** (~ 55-60 %
historiquement en NBA). Un modèle qui ne bat pas cette baseline n'apporte rien.

---

## Stack technique

| Rôle | Outil | Pourquoi |
|------|-------|----------|
| Récupération des données | `nba_api` | Wrapper officiel-communautaire des endpoints `stats.nba.com` |
| Manipulation des données | `pandas` / `numpy` | Jointures temporelles, fenêtres glissantes, formule des possessions |
| Cache disque | `pyarrow` (parquet) | Format colonne typé et compressé → évite de re-télécharger |
| Modèle | `scikit-learn` (`LogisticRegression`) | Classifieur linéaire interprétable + métriques + pipeline de scaling |
| Visualisation | `matplotlib` | Matrice de confusion, courbe ROC sauvegardées dans `reports/` |
| Exploration | `jupyter` | Analyse exploratoire commentée |

---

## Approche des données (choix clés)

- **Une seule requête `LeagueGameFinder` par saison** (et par type : saison régulière + playoffs).
  Chaque match revient en **deux lignes** (une par équipe). En les **auto-joignant sur `GAME_ID`**,
  on reconstitue pour chaque équipe : les points encaissés, le flag domicile/extérieur, et une
  **estimation des possessions** — donc un **defensive rating par match**, sans avoir à appeler
  ~12 000 fois l'endpoint box-score avancé. Rapide (~20 requêtes au total) et sans fuite de données.
- **Possessions estimées** par la formule standard :
  `POSS = FGA − OREB + TOV + 0.44 × FTA`, moyennée entre les deux équipes pour stabiliser.
  `DEF_RATING = 100 × points_encaissés / possessions`.
- **Rate-limiting** : `time.sleep` entre chaque appel + retries avec back-off, car `stats.nba.com`
  throttle / time-out facilement.
- **Cache parquet** dans `data/raw/` : un fichier par `(saison, type)`. Un second passage ne
  re-télécharge rien (sauf `--force`).

> Les *features* (rating défensif glissant, repos, back-to-back, forme) sont construites dans
> `features.py` à partir de ces blocs bruts, en **ne regardant que les matchs passés** de chaque équipe.

---

## Installation

```bash
# 1. Cloner puis se placer dans le repo
cd nba-game-outcome-predictor

# 2. Environnement virtuel (recommandé)
python3 -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate

# 3. Dépendances
pip install -r requirements.txt
```

> `nba_api` interroge `stats.nba.com`, qui peut bloquer certaines IP (data centers, CI, certains
> VPN). En cas d'erreurs de timeout systématiques, lancer l'ingestion depuis une connexion
> résidentielle. Le cache parquet permet ensuite de travailler hors-ligne.

---

## Utilisation

```bash
# 1. Télécharger ~10 saisons (long la 1re fois à cause du rate-limiting ; instantané ensuite)
python src/data_ingestion.py                 # tout le périmètre par défaut
python src/data_ingestion.py --seasons 2022-23 2023-24   # un sous-ensemble pour tester
python src/data_ingestion.py --force         # ignore le cache et re-télécharge

# 2. Construire les features (sortie : data/processed/features.parquet)
python src/features.py

# 3. Entraîner (split temporel) + comparer à la baseline
python src/train.py

# 4. Évaluer : accuracy, ROC-AUC, log-loss, matrice de confusion → reports/figures/
python src/evaluate.py

# 5. (Optionnel) Interface visuelle interactive
streamlit run app.py
```

### Interface visuelle (Streamlit)

`app.py` lance un tableau de bord dans le navigateur (`http://localhost:8501`) :

1. **Vue d'ensemble** — nb de matchs, saisons, avantage du terrain par saison.
2. **Performance** — tableau baseline vs régression logistique (depuis `reports/metrics_*.json`).
3. **Figures** — matrice de confusion + courbe ROC.
4. **Poids du modèle** — coefficients standardisés en bar chart.
5. **Prédicteur interactif** — choisis l'équipe à domicile et l'extérieur (+ jours de
   repos) → **probabilité de victoire** prédite en direct par le modèle entraîné.

```bash
source .venv/bin/activate
pip install streamlit        # si nécessaire
streamlit run app.py
```

> Prérequis : avoir exécuté les étapes 1→4 (l'app a besoin de `models/logreg.joblib`,
> des données et des figures). L'app ne ré-entraîne rien : elle charge le modèle déjà appris.

---

## Résultats

Résultats **réels** sur ~10 saisons (12 649 matchs après nettoyage) — **entraînement
2014-15 → 2022-23** (11 352 matchs), **test sur la saison 2023-24** (1 297 matchs),
split strictement temporel.

| Modèle | Accuracy | ROC-AUC | Log-loss |
|--------|:--------:|:-------:|:--------:|
| Baseline "domicile gagne toujours" | 54.6 % | 0.500 | 0.693 |
| **Régression logistique** | **62.6 %** | **0.653** | **0.654** |

→ **+8.0 points d'accuracy** sur la baseline, avec un ROC-AUC de 0.653 (la baseline,
qui prédit toujours la même classe, est à 0.5 par construction). Scores volontairement
**honnêtes** : pas de fuite temporelle, pas de tuning agressif.

**Lecture des coefficients** (features standardisées → directement comparables ;
signe = sens de l'effet sur P(victoire domicile). Intercept = **+0.311**, qui capte
l'avantage du terrain) :

| Feature | Coefficient | Interprétation |
|---------|:-----------:|----------------|
| `off_rating_diff_5` | **+0.380** | Mieux attaquer que l'adversaire sur 5 matchs = signal n°1 |
| `def_rating_diff_5` | **−0.276** | Rating défensif plus **bas** (meilleure défense) → plus de victoires |
| `rest_days_diff` | +0.087 | Être plus reposé que l'adversaire aide, à la marge |
| `form_5_diff` | +0.063 | La forme récente (% victoires) joue un peu |
| `back_to_back_diff` | ≈ 0.000 | Colinéaire avec `rest_days_diff` (0 j de repos) → signal absorbé par lui |
| `home_advantage` | 0.000 | Constante → effet entièrement porté par l'intercept (+0.311) |

Figures générées (`reports/figures/`) :

- `confusion_matrix.png` — matrice de confusion (sur 2023-24 : TN=258, FP=331, FN=154, TP=554)
- `roc_curve.png` — courbe ROC avec AUC

> Les chiffres ci-dessus se régénèrent à l'identique avec `train.py` + `evaluate.py`
> (les métriques brutes sont aussi écrites dans `reports/metrics_train.json` et
> `reports/metrics_eval.json`).

---

## Structure du dépôt

```
nba-game-outcome-predictor/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py                     # interface visuelle Streamlit (dashboard + prédicteur)
├── data/                      # généré, non versionné
│   ├── raw/                   # cache parquet par saison (sortie de data_ingestion.py)
│   └── processed/             # dataset de features (sortie de features.py)
├── models/                    # généré : logreg.joblib (sortie de train.py)
├── reports/
│   ├── metrics_train.json     # métriques + coefficients (sortie de train.py)
│   ├── metrics_eval.json      # métriques de test (sortie de evaluate.py)
│   └── figures/               # matrice de confusion, ROC (sortie de evaluate.py)
├── notebooks/
│   └── exploration.ipynb      # analyse exploratoire commentée
└── src/
    ├── data_ingestion.py      # téléchargement nba_api + rate-limiting + cache parquet
    ├── features.py            # rating défensif, repos, back-to-back, forme, avantage domicile
    ├── train.py               # split TEMPOREL, régression logistique, vs baseline
    └── evaluate.py            # accuracy, ROC-AUC, log-loss, matrice de confusion → figures
```

---

## Limites & pistes

- `stats.nba.com` n'est pas une API officielle stable : endpoints et throttling peuvent changer.
- Modèle linéaire volontairement simple → pas d'interactions ni d'effet joueur (blessures, repos
  ciblé des stars). Pistes : gradient boosting, Elo, cotes de marché comme baseline plus dure.
- Les possessions sont **estimées** (pas la donnée play-by-play exacte).
