"""
data_ingestion.py
=================

Télécharge ~10 saisons de résultats de matchs NBA (saison régulière + playoffs)
via `nba_api`, gère le rate-limiting de `stats.nba.com`, met en cache le résultat
brut en parquet dans ``data/raw/`` et produit un dataset consolidé au niveau
*équipe × match*.

Choix techniques (le "pourquoi")
--------------------------------
1. **Endpoint : `LeagueGameFinder`.**
   C'est l'endpoint le plus économe pour récupérer TOUS les box-scores d'équipe
   d'une saison en **une seule requête**. Chaque match y figure en **deux lignes**
   (une par équipe). En auto-joignant ces lignes sur ``GAME_ID`` on obtient, pour
   chaque équipe et chaque match : les points encaissés, le flag domicile/extérieur
   et de quoi estimer les possessions — donc un *defensive rating par match* sans
   appeler ~12 000 fois l'endpoint box-score avancé.

2. **Possessions estimées** (formule "Dean Oliver" classique) :
   ``POSS = FGA - OREB + TOV + 0.44 * FTA`` ; on moyenne les possessions des deux
   équipes (le rythme d'un match est partagé) pour un estimateur plus stable.
   ``DEF_RATING = 100 * points_encaissés / possessions``.

3. **Rate-limiting.** `stats.nba.com` throttle agressivement : on insère un
   ``time.sleep`` entre chaque appel et on retente avec back-off exponentiel sur
   timeout / erreur réseau.

4. **Cache parquet.** Un fichier brut par ``(saison, type)`` dans ``data/raw/``.
   Un second lancement ne re-télécharge rien (sauf ``--force``). Parquet = format
   colonne typé et compressé, lecture quasi instantanée.

Sortie
------
- ``data/raw/games_<season>_<type>.parquet`` : brut par saison (1 ligne / équipe / match).
- ``data/raw/games_team_level.parquet``      : consolidé, enrichi (opp. points,
  possessions, off/def rating par match, flag domicile). C'est l'entrée de ``features.py``.

Usage
-----
    python src/data_ingestion.py                       # périmètre par défaut
    python src/data_ingestion.py --seasons 2022-23 2023-24
    python src/data_ingestion.py --no-playoffs
    python src/data_ingestion.py --force               # ignore le cache
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

# `nba_api` est importé paresseusement dans `_fetch_one` pour que le module reste
# importable (tests, lecture du cache) même si le package n'est pas installé.

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("data_ingestion")

# Racine du projet = parent de src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# 10 saisons par défaut : 2014-15 → 2023-24 (formatées comme l'attend stats.nba.com).
DEFAULT_SEASONS: list[str] = [f"{y}-{str(y + 1)[-2:]}" for y in range(2014, 2024)]

# Libellés exacts attendus par l'endpoint pour `season_type_nullable`.
SEASON_TYPE_REGULAR = "Regular Season"
SEASON_TYPE_PLAYOFFS = "Playoffs"

# Rate-limiting / robustesse
SLEEP_BETWEEN_CALLS_S = 1.5   # pause systématique entre deux requêtes réussies
REQUEST_TIMEOUT_S = 60        # timeout passé à l'endpoint nba_api
MAX_RETRIES = 4               # nb de tentatives par requête
BACKOFF_BASE_S = 2.0          # délai = BACKOFF_BASE ** tentative

# Colonnes box-score d'équipe que l'on conserve depuis LeagueGameFinder.
KEEP_COLS = [
    "SEASON_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME",
    "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "PTS",
    "FGA", "FTA", "OREB", "TOV",  # nécessaires à l'estimation des possessions
]


# --------------------------------------------------------------------------- #
# Téléchargement (1 requête = 1 saison × 1 type)
# --------------------------------------------------------------------------- #

def _fetch_one(season: str, season_type: str) -> pd.DataFrame:
    """Télécharge tous les box-scores d'équipe d'une (saison, type) en 1 requête.

    Retries avec back-off exponentiel : `stats.nba.com` renvoie fréquemment des
    timeouts ou des réponses vides quand on enchaîne les appels trop vite.
    """
    # Import paresseux : permet d'importer ce module sans nba_api installé.
    from nba_api.stats.endpoints import leaguegamefinder

    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("Téléchargement %s | %s (essai %d/%d)…",
                        season, season_type, attempt, MAX_RETRIES)
            finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable=season_type,
                league_id_nullable="00",          # 00 = NBA (≠ G-League / WNBA)
                timeout=REQUEST_TIMEOUT_S,
            )
            df = finder.get_data_frames()[0]
            if df.empty:
                # Une saison sans playoffs (saison en cours) renvoie un df vide :
                # ce n'est pas une erreur, on remonte le df vide tel quel.
                logger.warning("Réponse vide pour %s | %s.", season, season_type)
            return df
        except Exception as err:  # noqa: BLE001 — on veut retenter sur tout échec réseau
            last_err = err
            wait = BACKOFF_BASE_S ** attempt
            logger.warning("Échec (%s). Nouvelle tentative dans %.1fs…", err, wait)
            time.sleep(wait)

    raise RuntimeError(
        f"Impossible de télécharger {season} | {season_type} "
        f"après {MAX_RETRIES} tentatives : {last_err}"
    )


def _cache_path(season: str, season_type: str) -> Path:
    slug = season_type.lower().replace(" ", "_")
    return RAW_DIR / f"games_{season}_{slug}.parquet"


def load_or_download(season: str, season_type: str, *, force: bool = False) -> pd.DataFrame:
    """Renvoie le brut d'une (saison, type), depuis le cache si présent."""
    path = _cache_path(season, season_type)
    if path.exists() and not force:
        logger.info("Cache hit : %s", path.name)
        return pd.read_parquet(path)

    df = _fetch_one(season, season_type)
    df = df[[c for c in KEEP_COLS if c in df.columns]].copy()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Mis en cache : %s (%d lignes)", path.name, len(df))

    # Pause de courtoisie uniquement après un vrai appel réseau (pas un cache hit).
    time.sleep(SLEEP_BETWEEN_CALLS_S)
    return df


# --------------------------------------------------------------------------- #
# Consolidation & enrichissement
# --------------------------------------------------------------------------- #

def _estimate_possessions(df: pd.DataFrame) -> pd.Series:
    """Possessions estimées d'une équipe sur un match (formule de Dean Oliver)."""
    return df["FGA"] - df["OREB"] + df["TOV"] + 0.44 * df["FTA"]


def build_team_level(raw: pd.DataFrame) -> pd.DataFrame:
    """Auto-jointure pour passer de "ligne/équipe" à "ligne/équipe + adversaire".

    Pour chaque ``GAME_ID`` il y a exactement 2 lignes. On joint la table sur
    elle-même par ``GAME_ID`` puis on retire l'auto-appariement (même équipe), ce
    qui rattache à chaque équipe les stats de son adversaire.
    """
    df = raw.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # Domicile vs extérieur : "TEAM vs. OPP" = domicile, "TEAM @ OPP" = extérieur.
    df["IS_HOME"] = df["MATCHUP"].str.contains("vs.", regex=False).astype(int)

    # Possessions de l'équipe sur ce match (avant jointure).
    df["POSS_TEAM"] = _estimate_possessions(df)

    # Colonnes de l'adversaire que l'on veut récupérer via la self-join.
    opp = df[["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "PTS", "POSS_TEAM"]].rename(
        columns={
            "TEAM_ID": "OPP_TEAM_ID",
            "TEAM_ABBREVIATION": "OPP_TEAM_ABBREVIATION",
            "PTS": "OPP_PTS",
            "POSS_TEAM": "OPP_POSS_TEAM",
        }
    )

    merged = df.merge(opp, on="GAME_ID", how="inner")
    # On retire les lignes où l'équipe est appariée avec elle-même.
    merged = merged[merged["TEAM_ID"] != merged["OPP_TEAM_ID"]].copy()

    # Garde-fou : chaque (GAME_ID, TEAM_ID) doit être unique après jointure.
    dup = merged.duplicated(subset=["GAME_ID", "TEAM_ID"]).sum()
    if dup:
        logger.warning("%d doublons (GAME_ID, TEAM_ID) après jointure.", dup)

    # Rythme du match : moyenne des possessions des deux équipes (estimateur stable).
    merged["GAME_POSS"] = 0.5 * (merged["POSS_TEAM"] + merged["OPP_POSS_TEAM"])

    # Ratings par match (×100 = par 100 possessions). Garde-fou division par 0.
    safe_poss = merged["GAME_POSS"].replace(0, pd.NA)
    merged["OFF_RATING"] = 100 * merged["PTS"] / safe_poss
    merged["DEF_RATING"] = 100 * merged["OPP_PTS"] / safe_poss

    # Cible : l'équipe a-t-elle gagné ce match ?
    merged["WON"] = (merged["WL"] == "W").astype(int)

    keep = [
        "SEASON_ID", "SEASON", "SEASON_TYPE", "GAME_ID", "GAME_DATE",
        "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME",
        "OPP_TEAM_ID", "OPP_TEAM_ABBREVIATION",
        "IS_HOME", "PTS", "OPP_PTS", "GAME_POSS",
        "OFF_RATING", "DEF_RATING", "WON",
    ]
    keep = [c for c in keep if c in merged.columns]
    return merged[keep].sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def ingest(
    seasons: list[str],
    *,
    include_playoffs: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """Télécharge toutes les (saison, type), consolide et sauvegarde le team-level."""
    season_types = [SEASON_TYPE_REGULAR]
    if include_playoffs:
        season_types.append(SEASON_TYPE_PLAYOFFS)

    frames: list[pd.DataFrame] = []
    for season in seasons:
        for season_type in season_types:
            df = load_or_download(season, season_type, force=force)
            if df.empty:
                continue
            # On annote la saison et le type (utiles pour le split temporel / l'EDA).
            df = df.assign(SEASON=season, SEASON_TYPE=season_type)
            frames.append(df)

    if not frames:
        raise RuntimeError("Aucune donnée récupérée — vérifie la connexion / l'accès à stats.nba.com.")

    raw_all = pd.concat(frames, ignore_index=True)
    logger.info("Brut consolidé : %d lignes équipe×match.", len(raw_all))

    team_level = build_team_level(raw_all)
    out_path = RAW_DIR / "games_team_level.parquet"
    team_level.to_parquet(out_path, index=False)
    logger.info("Écrit : %s (%d lignes, %d matchs).",
                out_path, len(team_level), team_level["GAME_ID"].nunique())
    return team_level


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingestion des données NBA via nba_api.")
    p.add_argument(
        "--seasons", nargs="+", default=DEFAULT_SEASONS,
        help="Saisons au format 'YYYY-YY' (def : 2014-15 … 2023-24).",
    )
    p.add_argument(
        "--no-playoffs", action="store_true",
        help="Saison régulière uniquement (par défaut les playoffs sont inclus).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Ignore le cache parquet et re-télécharge.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logger.info("Saisons : %s | playoffs : %s | force : %s",
                ", ".join(args.seasons), not args.no_playoffs, args.force)
    df = ingest(args.seasons, include_playoffs=not args.no_playoffs, force=args.force)

    # Petit récapitulatif lisible en fin de run.
    logger.info("Aperçu (5 premières lignes) :\n%s",
                df.head().to_string(index=False))


if __name__ == "__main__":
    main()
