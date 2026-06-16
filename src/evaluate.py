"""
evaluate.py
===========

Évalue le modèle entraîné par ``train.py`` sur la **même saison de test** (split
temporel identique) et produit :

- les métriques : accuracy, ROC-AUC, log-loss, matrice de confusion (console + JSON) ;
- deux figures dans ``reports/figures/`` :
    * ``confusion_matrix.png`` ;
    * ``roc_curve.png`` (avec l'AUC et la diagonale du hasard).

On **réutilise** ``temporal_split`` de ``train.py`` : impossible que le découpage
diverge entre entraînement et évaluation (une seule source de vérité).

Usage
-----
    python src/evaluate.py
    python src/evaluate.py --test-season 2023-24
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # backend non-interactif → écrit des PNG sans serveur graphique
import matplotlib.pyplot as plt  # noqa: E402  (après le choix du backend)
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    roc_curve,
)

from features import FEATURE_COLUMNS, TARGET_COLUMN  # noqa: E402
from train import temporal_split  # noqa: E402  (réutilise le même split)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "logreg.joblib"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics_eval.json"


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Path:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(
        cm, display_labels=["Domicile perd", "Domicile gagne"]
    ).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Matrice de confusion — saison test")
    fig.tight_layout()
    path = FIG_DIR / "confusion_matrix.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _plot_roc(y_true: np.ndarray, proba: np.ndarray, auc: float) -> Path:
    fpr, tpr, _ = roc_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, lw=2, label=f"Régression log. (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", label="Hasard (AUC = 0.5)")
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Courbe ROC — saison test")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = FIG_DIR / "roc_curve.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Évaluation
# --------------------------------------------------------------------------- #

def evaluate(df: pd.DataFrame, test_season: str | None) -> dict:
    model = joblib.load(MODEL_PATH)
    _, test_df = temporal_split(df, test_season)

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].to_numpy()

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    auc = float(roc_auc_score(y_test, proba))
    cm = confusion_matrix(y_test, pred)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "roc_auc": auc,
        "log_loss": float(log_loss(y_test, proba)),
        "confusion_matrix": cm.tolist(),
        "n_test": int(len(test_df)),
        "test_seasons": sorted(test_df["SEASON"].unique().tolist()),
    }

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cm_path = _plot_confusion_matrix(y_test, pred)
    roc_path = _plot_roc(y_test, proba, auc)

    logger.info("Accuracy : %.3f | ROC-AUC : %.3f | Log-loss : %.3f",
                metrics["accuracy"], metrics["roc_auc"], metrics["log_loss"])
    logger.info("Matrice de confusion [[TN, FP], [FN, TP]] :\n%s", cm)
    logger.info("Figures : %s | %s", cm_path.name, roc_path.name)

    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    logger.info("Métriques → %s", METRICS_PATH)
    return metrics


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Évaluation du modèle + figures.")
    p.add_argument("--test-season", default=None,
                   help="Saison de test (doit correspondre à l'entraînement).")
    return p.parse_args()


def main() -> None:
    for path, hint in [
        (FEATURES_PATH, "python src/features.py"),
        (MODEL_PATH, "python src/train.py"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{path} introuvable — lance d'abord : {hint}")

    args = _parse_args()
    df = pd.read_parquet(FEATURES_PATH)
    evaluate(df, args.test_season)


if __name__ == "__main__":
    main()
