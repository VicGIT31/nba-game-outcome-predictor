"""
train.py
========

Entraîne une **régression logistique** à prédire la victoire de l'équipe à domicile,
avec un **split temporel** (on apprend sur le passé, on teste sur le futur), et la
compare à la **baseline "domicile gagne toujours"**.

Pourquoi un split TEMPOREL et pas aléatoire ?
---------------------------------------------
Un ``train_test_split`` aléatoire mélangerait passé et futur : le modèle pourrait
« voir » des matchs de mars pour en prédire de janvier. En production on prédit
toujours le futur à partir du passé → on **réplique** cette contrainte en triant
par date et en coupant à une frontière temporelle. C'est plus dur, mais honnête.

Pourquoi un `Pipeline` avec `StandardScaler` ?
----------------------------------------------
Les features ont des échelles très différentes (jours de repos ~ unités, defensive
rating ~ 100). La régression logistique régularisée (L2) est sensible à l'échelle.
Le scaler est **fit sur le train uniquement** (via le Pipeline) → pas de fuite des
statistiques du test.

Sorties
-------
- ``models/logreg.joblib``        : pipeline entraîné (réutilisé par evaluate.py).
- ``reports/metrics_train.json``  : métriques train/test + baseline + coefficients.

Usage
-----
    python src/train.py
    python src/train.py --test-season 2023-24   # frontière de split explicite
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLUMNS, TARGET_COLUMN  # source de vérité des colonnes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "logreg.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics_train.json"

# Par défaut, la dernière saison du dataset sert de test (le reste = train).
DEFAULT_TEST_SEASON: str | None = None


# --------------------------------------------------------------------------- #
# Split temporel
# --------------------------------------------------------------------------- #

def temporal_split(
    df: pd.DataFrame, test_season: str | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Coupe le dataset en train (passé) / test (futur) sans mélange.

    - Si ``test_season`` est donné : cette saison (et au-delà) = test.
    - Sinon : la **dernière saison** présente dans les données = test.
    """
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    seasons = sorted(df["SEASON"].unique())

    if test_season is None:
        test_season = seasons[-1]
    if test_season not in seasons:
        raise ValueError(f"Saison de test '{test_season}' absente. Dispo : {seasons}")

    cutoff_idx = seasons.index(test_season)
    train_seasons = set(seasons[:cutoff_idx])
    test_seasons = set(seasons[cutoff_idx:])

    train = df[df["SEASON"].isin(train_seasons)].copy()
    test = df[df["SEASON"].isin(test_seasons)].copy()

    logger.info("Split temporel → train : %s (%d matchs) | test : %s (%d matchs)",
                sorted(train_seasons), len(train), sorted(test_seasons), len(test))
    return train, test


# --------------------------------------------------------------------------- #
# Métriques
# --------------------------------------------------------------------------- #

def _evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Accuracy / ROC-AUC / log-loss à partir des probabilités prédites."""
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "log_loss": float(log_loss(y_true, proba)),
    }


def _baseline_metrics(y_true: np.ndarray) -> dict:
    """Baseline "domicile gagne toujours" : prédit 1 partout, proba constante.

    ROC-AUC = 0.5 par construction (aucun pouvoir de discrimination) ; on calcule
    l'accuracy réelle = simplement le taux de victoires à domicile sur le test.
    """
    pred = np.ones_like(y_true)
    proba = np.full_like(y_true, 0.5, dtype=float)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "roc_auc": 0.5,
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1])),
    }


# --------------------------------------------------------------------------- #
# Entraînement
# --------------------------------------------------------------------------- #

def train(df: pd.DataFrame, test_season: str | None) -> dict:
    train_df, test_df = temporal_split(df, test_season)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN].to_numpy()
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN].to_numpy()

    # `home_advantage` est constante (=1) → on la laisse au modèle qui l'absorbe
    # via l'intercept ; StandardScaler la met à 0, son effet passe dans le biais.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0)),
    ])
    model.fit(X_train, y_train)

    proba_train = model.predict_proba(X_train)[:, 1]
    proba_test = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": _evaluate(y_test, proba_test),
        "model_train": _evaluate(y_train, proba_train),  # détecte le sur-apprentissage
        "baseline_home_always": _baseline_metrics(y_test),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "test_seasons": sorted(test_df["SEASON"].unique().tolist()),
    }

    # Coefficients lisibles (sur features standardisées → comparables entre elles).
    coefs = dict(zip(FEATURE_COLUMNS, model.named_steps["clf"].coef_[0].tolist()))
    metrics["coefficients"] = coefs
    metrics["intercept"] = float(model.named_steps["clf"].intercept_[0])

    _log_summary(metrics)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    logger.info("Modèle → %s | métriques → %s", MODEL_PATH, METRICS_PATH)

    return metrics


def _log_summary(m: dict) -> None:
    b, mod = m["baseline_home_always"], m["model"]
    logger.info("=" * 56)
    logger.info("%-22s | %8s | %8s | %8s", "", "accuracy", "roc_auc", "log_loss")
    logger.info("%-22s | %8.3f | %8s | %8s",
                "baseline (domicile)", b["accuracy"], "0.500", f"{b['log_loss']:.3f}")
    logger.info("%-22s | %8.3f | %8.3f | %8.3f",
                "régression log.", mod["accuracy"], mod["roc_auc"], mod["log_loss"])
    gain = (mod["accuracy"] - b["accuracy"]) * 100
    logger.info("Gain d'accuracy vs baseline : %+.2f points", gain)
    logger.info("Coefficients (features standardisées) :")
    for name, c in sorted(m["coefficients"].items(), key=lambda kv: -abs(kv[1])):
        logger.info("   %-20s %+.4f", name, c)
    logger.info("=" * 56)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entraînement régression logistique (split temporel).")
    p.add_argument("--test-season", default=DEFAULT_TEST_SEASON,
                   help="Saison (incluse) qui démarre le test. Défaut : dernière saison.")
    return p.parse_args()


def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"{FEATURES_PATH} introuvable — lance d'abord : python src/features.py"
        )
    args = _parse_args()
    df = pd.read_parquet(FEATURES_PATH)
    logger.info("Dataset features : %d matchs sur saisons %s.",
                len(df), sorted(df["SEASON"].unique()))
    train(df, args.test_season)


if __name__ == "__main__":
    main()
