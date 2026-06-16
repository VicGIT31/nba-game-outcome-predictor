"""
app.py — Interface visuelle Streamlit
=====================================

Démo interactive du NBA Game Outcome Predictor :

1. **Vue d'ensemble** des données (nb de matchs, saisons, avantage du terrain).
2. **Performance** du modèle vs baseline (lue depuis reports/metrics_*.json).
3. **Figures** (matrice de confusion, ROC) générées par evaluate.py.
4. **Poids du modèle** (coefficients standardisés) en bar chart.
5. **Prédicteur interactif** : on choisit l'équipe à domicile et l'extérieur,
   l'app reconstruit leurs stats de forme récentes et affiche la **probabilité
   de victoire** prédite par la régression logistique entraînée.

Lancement :
    source .venv/bin/activate
    pip install streamlit            # si pas déjà fait
    streamlit run app.py

Prérequis : avoir lancé data_ingestion.py → features.py → train.py → evaluate.py
(le modèle models/logreg.joblib et les données doivent exister).

NB : les fonctions de chargement / prédiction sont volontairement **pures** (sans
appel `st.*`) pour pouvoir être testées sans serveur Streamlit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # pour importer features.py

from features import FEATURE_COLUMNS, ROLL_WINDOW  # noqa: E402

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "games_team_level.parquet"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "logreg.joblib"
METRICS_TRAIN = PROJECT_ROOT / "reports" / "metrics_train.json"
METRICS_EVAL = PROJECT_ROOT / "reports" / "metrics_eval.json"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"


# --------------------------------------------------------------------------- #
# Logique pure (chargement + prédiction) — pas de dépendance à Streamlit
# --------------------------------------------------------------------------- #

def compute_team_profiles(team_level: pd.DataFrame, window: int = ROLL_WINDOW) -> pd.DataFrame:
    """Profil de forme récent par équipe = moyenne sur ses `window` derniers matchs.

    Sert d'estimation "à l'instant T" pour le prédicteur : on prend les derniers
    matchs connus de chaque équipe (toutes saisons confondues, le plus récent en base).
    """
    df = team_level.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE")

    rows = []
    for team, g in df.groupby("TEAM_ABBREVIATION"):
        last = g.tail(window)
        rows.append({
            "team": team,
            "off_rating_5": last["OFF_RATING"].mean(),
            "def_rating_5": last["DEF_RATING"].mean(),
            "form_5": last["WON"].mean(),
            "last_game": g["GAME_DATE"].max().date().isoformat(),
        })
    return pd.DataFrame(rows).set_index("team").sort_index()


def build_feature_row(
    home: pd.Series, away: pd.Series, home_rest: int, away_rest: int
) -> pd.DataFrame:
    """Construit la ligne de features (diff domicile − extérieur) pour la prédiction."""
    row = {
        "home_advantage": 1,
        "rest_days_diff": home_rest - away_rest,
        "back_to_back_diff": int(home_rest == 0) - int(away_rest == 0),
        "form_5_diff": home["form_5"] - away["form_5"],
        "def_rating_diff_5": home["def_rating_5"] - away["def_rating_5"],
        "off_rating_diff_5": home["off_rating_5"] - away["off_rating_5"],
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]


def predict_home_win_proba(model, feature_row: pd.DataFrame) -> float:
    """Probabilité que l'équipe à domicile gagne."""
    return float(model.predict_proba(feature_row)[:, 1][0])


# --------------------------------------------------------------------------- #
# Cache Streamlit
# --------------------------------------------------------------------------- #

@st.cache_resource
def _load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def _load_team_level() -> pd.DataFrame:
    return pd.read_parquet(RAW_PATH)


@st.cache_data
def _load_features() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_PATH)


@st.cache_data
def _load_metrics() -> tuple[dict, dict]:
    train = json.loads(METRICS_TRAIN.read_text()) if METRICS_TRAIN.exists() else {}
    eval_ = json.loads(METRICS_EVAL.read_text()) if METRICS_EVAL.exists() else {}
    return train, eval_


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

def _require_artifacts() -> bool:
    missing = [p.name for p in (RAW_PATH, FEATURES_PATH, MODEL_PATH) if not p.exists()]
    if missing:
        st.error(
            "Artefacts manquants : " + ", ".join(missing) + ".\n\n"
            "Lance d'abord :\n"
            "```\npython src/data_ingestion.py\npython src/features.py\n"
            "python src/train.py\npython src/evaluate.py\n```"
        )
        return False
    return True


def main() -> None:
    st.set_page_config(page_title="NBA Game Outcome Predictor", page_icon="🏀", layout="wide")
    st.title("🏀 NBA Game Outcome Predictor")
    st.caption(
        "Régression logistique entraînée sur ~10 saisons (nba_api) — split temporel, "
        "features sans fuite. Prédit la probabilité de victoire de l'équipe à domicile."
    )

    if not _require_artifacts():
        st.stop()

    model = _load_model()
    team_level = _load_team_level()
    features = _load_features()
    metrics_train, metrics_eval = _load_metrics()
    profiles = compute_team_profiles(team_level)

    # ---- Vue d'ensemble ----
    st.header("1. Vue d'ensemble des données")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matchs (features)", f"{len(features):,}".replace(",", " "))
    c2.metric("Saisons", features["SEASON"].nunique())
    c3.metric("Équipes", profiles.shape[0])
    c4.metric("Victoires à domicile", f"{features['home_win'].mean():.1%}")

    by_season = features.groupby("SEASON")["home_win"].mean()
    st.bar_chart(by_season, height=260, y_label="Taux de victoire domicile")
    st.caption("Avantage du terrain par saison — noter l'effondrement en 2020-21 (COVID, arènes vides).")

    # ---- Performance ----
    st.header("2. Performance (saison test, split temporel)")
    if metrics_train:
        base = metrics_train["baseline_home_always"]
        mod = metrics_train["model"]
        perf = pd.DataFrame({
            "Accuracy": [base["accuracy"], mod["accuracy"]],
            "ROC-AUC": [base["roc_auc"], mod["roc_auc"]],
            "Log-loss": [base["log_loss"], mod["log_loss"]],
        }, index=["Baseline (domicile gagne)", "Régression logistique"])
        st.dataframe(perf.style.format("{:.3f}"), use_container_width=True)
        gain = (mod["accuracy"] - base["accuracy"]) * 100
        st.success(f"Le modèle bat la baseline de **{gain:+.1f} points** d'accuracy "
                   f"(test : {', '.join(metrics_train.get('test_seasons', []))}).")

    # ---- Figures ----
    st.header("3. Figures d'évaluation")
    f1, f2 = st.columns(2)
    cm, roc = FIG_DIR / "confusion_matrix.png", FIG_DIR / "roc_curve.png"
    if cm.exists():
        f1.image(str(cm), caption="Matrice de confusion", use_container_width=True)
    if roc.exists():
        f2.image(str(roc), caption="Courbe ROC", use_container_width=True)

    # ---- Coefficients ----
    st.header("4. Poids du modèle (coefficients standardisés)")
    if metrics_train.get("coefficients"):
        coefs = pd.Series(metrics_train["coefficients"]).sort_values()
        st.bar_chart(coefs, height=300, x_label="coefficient")
        st.caption("Signe = sens de l'effet sur P(victoire domicile). "
                   "off/def rating récents dominent ; back-to-back ≈ 0 (colinéaire au repos).")

    # ---- Prédicteur interactif ----
    st.header("5. Prédire un match")
    teams = profiles.index.tolist()
    p1, p2 = st.columns(2)
    home = p1.selectbox("Équipe à domicile", teams, index=teams.index("LAL") if "LAL" in teams else 0)
    away = p2.selectbox("Équipe à l'extérieur", teams, index=teams.index("BOS") if "BOS" in teams else 1)
    home_rest = p1.slider("Jours de repos — domicile", 0, 7, 2)
    away_rest = p2.slider("Jours de repos — extérieur", 0, 7, 2)

    if home == away:
        st.warning("Choisis deux équipes différentes.")
        st.stop()

    row = build_feature_row(profiles.loc[home], profiles.loc[away], home_rest, away_rest)
    proba_home = predict_home_win_proba(model, row)

    r1, r2 = st.columns(2)
    r1.metric(f"P(victoire {home}, domicile)", f"{proba_home:.1%}")
    r2.metric(f"P(victoire {away}, extérieur)", f"{1 - proba_home:.1%}")
    winner = home if proba_home >= 0.5 else away
    st.progress(proba_home, text=f"Favori : **{winner}**")

    with st.expander("Détail des features utilisées pour cette prédiction"):
        st.dataframe(row.T.rename(columns={0: "valeur"}), use_container_width=True)
        st.caption(f"Profils calculés sur les {ROLL_WINDOW} derniers matchs en base "
                   f"(domicile : dernier match {profiles.loc[home, 'last_game']}, "
                   f"extérieur : {profiles.loc[away, 'last_game']}).")


if __name__ == "__main__":
    main()
