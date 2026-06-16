"""
features.py
===========

Construit le dataset d'entraînement à partir du brut *équipe × match*
(``data/raw/games_team_level.parquet`` produit par ``data_ingestion.py``) et le
sauvegarde dans ``data/processed/features.parquet``.

Philosophie : **aucune fuite temporelle.**
----------------------------------------------------------------------
Une feature ne doit utiliser **que l'information disponible AVANT le coup d'envoi**.
Concrètement, pour chaque équipe, toute statistique « de forme » est calculée sur
ses matchs **strictement précédents** :

- on trie par date,
- on calcule une moyenne glissante,
- puis on la **décale d'un cran** (``shift(1)``) pour exclure le match courant.

Sans ce ``shift``, le résultat du match courant fuiterait dans ses propres features
et l'évaluation serait trop optimiste (erreur classique et éliminatoire en entretien).

Features produites (différence domicile − extérieur)
----------------------------------------------------
- ``home_advantage``      : constante = 1 (capte l'avantage du terrain via l'intercept/coef).
- ``rest_days_diff``      : jours de repos (domicile) − jours de repos (extérieur).
- ``back_to_back_diff``   : indicateur back-to-back (0 j de repos) domicile − extérieur.
- ``form_5_diff``         : % de victoires sur les 5 derniers matchs, domicile − extérieur.
- ``def_rating_diff_5``   : defensive rating moyen sur 5 matchs (plus bas = mieux), dom − ext.
- ``off_rating_diff_5``   : offensive rating moyen sur 5 matchs, dom − ext (bonus, même logique).

Cible
-----
- ``home_win`` : 1 si l'équipe à domicile gagne, sinon 0. C'est aussi exactement la
  prédiction de la **baseline "domicile gagne toujours"**, d'où ce point de vue.

Usage
-----
    python src/features.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("features")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "games_team_level.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"

ROLL_WINDOW = 5  # taille de la fenêtre "forme récente"

# Liste centralisée des features → réutilisée telle quelle par train.py / evaluate.py.
FEATURE_COLUMNS = [
    "home_advantage",
    "rest_days_diff",
    "back_to_back_diff",
    "form_5_diff",
    "def_rating_diff_5",
    "off_rating_diff_5",
]
TARGET_COLUMN = "home_win"


# --------------------------------------------------------------------------- #
# Étape 1 — features "par équipe" (décalées dans le temps, sans fuite)
# --------------------------------------------------------------------------- #

def _team_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute, pour chaque ligne équipe×match, ses features pré-match.

    Toutes les statistiques sont calculées sur les matchs PASSÉS de l'équipe
    (``shift(1)`` après tri chronologique), saison par saison pour ne pas faire
    déborder la forme d'une saison sur la suivante.
    """
    df = df.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).copy()

    # --- Repos & back-to-back ------------------------------------------------
    # Jours depuis le match précédent de la MÊME équipe, DANS la même saison.
    prev_date = df.groupby(["TEAM_ID", "SEASON"])["GAME_DATE"].shift(1)
    df["rest_days"] = (df["GAME_DATE"] - prev_date).dt.days
    # 1er match de saison : pas de "repos" défini → valeur neutre (médiane ~2-3 j).
    df["rest_days"] = df["rest_days"].fillna(3).clip(upper=10)
    # Back-to-back = a joué la veille (0 jour de repos).
    df["back_to_back"] = (df["rest_days"] == 0).astype(int)

    # --- Forme & ratings glissants (shiftés = strictement passés) ------------
    grp = df.groupby(["TEAM_ID", "SEASON"], group_keys=False)

    def _shifted_roll(col: str) -> pd.Series:
        # moyenne sur les ROLL_WINDOW matchs PRÉCÉDENTS (min_periods=1 pour le début
        # de saison), puis shift(1) pour exclure le match courant.
        return grp[col].apply(
            lambda s: s.rolling(ROLL_WINDOW, min_periods=1).mean().shift(1)
        )

    df["form_5"] = _shifted_roll("WON")          # % victoires sur 5 derniers
    df["def_rating_5"] = _shifted_roll("DEF_RATING")
    df["off_rating_5"] = _shifted_roll("OFF_RATING")

    return df


# --------------------------------------------------------------------------- #
# Étape 2 — repli au niveau match (domicile vs extérieur)
# --------------------------------------------------------------------------- #

def _pivot_to_game_level(team_df: pd.DataFrame) -> pd.DataFrame:
    """Repasse de 2 lignes/match (par équipe) à 1 ligne/match (domicile vs ext.).

    On sépare les lignes domicile et extérieur puis on les rejoint sur ``GAME_ID``.
    Les features finales sont les **différences** domicile − extérieur.
    """
    cols = [
        "GAME_ID", "GAME_DATE", "SEASON", "SEASON_TYPE",
        "TEAM_ABBREVIATION", "WON",
        "rest_days", "back_to_back", "form_5", "def_rating_5", "off_rating_5",
    ]
    home = team_df.loc[team_df["IS_HOME"] == 1, cols].add_prefix("home_")
    away = team_df.loc[team_df["IS_HOME"] == 0, cols].add_prefix("away_")

    games = home.merge(
        away, left_on="home_GAME_ID", right_on="away_GAME_ID", how="inner"
    )

    out = pd.DataFrame({
        "GAME_ID": games["home_GAME_ID"],
        "GAME_DATE": games["home_GAME_DATE"],
        "SEASON": games["home_SEASON"],
        "SEASON_TYPE": games["home_SEASON_TYPE"],
        "home_team": games["home_TEAM_ABBREVIATION"],
        "away_team": games["away_TEAM_ABBREVIATION"],
        # Cible = baseline "domicile gagne toujours".
        TARGET_COLUMN: games["home_WON"].astype(int),
        # Features = différences domicile − extérieur.
        "home_advantage": 1,
        "rest_days_diff": games["home_rest_days"] - games["away_rest_days"],
        "back_to_back_diff": games["home_back_to_back"] - games["away_back_to_back"],
        "form_5_diff": games["home_form_5"] - games["away_form_5"],
        "def_rating_diff_5": games["home_def_rating_5"] - games["away_def_rating_5"],
        "off_rating_diff_5": games["home_off_rating_5"] - games["away_off_rating_5"],
    })
    return out.sort_values("GAME_DATE").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def build_features(team_df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline complet : features par équipe → niveau match → nettoyage."""
    team_df = team_df.copy()
    team_df["GAME_DATE"] = pd.to_datetime(team_df["GAME_DATE"])

    enriched = _team_rolling_features(team_df)
    games = _pivot_to_game_level(enriched)

    before = len(games)
    # Les tout premiers matchs de chaque saison ont des ratings glissants à NaN
    # (aucun match passé) → on les retire pour ne pas inventer de valeurs.
    games = games.dropna(subset=["form_5_diff", "def_rating_diff_5", "off_rating_diff_5"])
    logger.info("Matchs retirés (début de saison, features incomplètes) : %d", before - len(games))

    return games.reset_index(drop=True)


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"{RAW_PATH} introuvable — lance d'abord : python src/data_ingestion.py"
        )

    team_df = pd.read_parquet(RAW_PATH)
    logger.info("Brut chargé : %d lignes équipe×match.", len(team_df))

    features = build_features(team_df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUT_PATH, index=False)

    logger.info("Écrit : %s (%d matchs, %d features).",
                OUT_PATH, len(features), len(FEATURE_COLUMNS))
    logger.info("Taux de victoire à domicile (cible) : %.3f", features[TARGET_COLUMN].mean())
    logger.info("Aperçu :\n%s", features.head().to_string(index=False))


if __name__ == "__main__":
    main()
