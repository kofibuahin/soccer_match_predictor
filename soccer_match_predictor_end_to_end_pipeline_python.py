"""
Soccer Match Predictor – End-to-End Pipeline (Python)
====================================================

End-to-end, resume-ready pipeline for predicting soccer match outcomes (H/D/A)
from publicly available data (football-data.co.uk). Includes data acquisition,
feature engineering (Elo, rolling form, odds), model training & evaluation,
probability calibration, artifact persistence, and a starter prediction API.

Usage (examples):
-----------------
# 1) Install deps (suggested)
#    pip install pandas numpy scikit-learn requests joblib pyarrow tqdm matplotlib

# 2) Run the script to download data for top leagues & train a model
#    python soccer_match_predictor_pipeline.py --start_season 2015 --end_season 2024 --leagues E0 SP1 I1 D1 F1

# 3) Save artifacts to ./artifacts/ (model.pkl, columns.json, metadata.json)

# 4) Predict for given fixtures (teams must match football-data.co.uk names)
#    python soccer_match_predictor_pipeline.py --predict 
#        --fixtures "2025-08-15,Manchester City,Arsenal" "2025-08-15,Barcelona,Real Madrid"

Notes:
------
- Seasons use football-data.co.uk scheme (e.g., 2015→"1516", 2024→"2425").
- League codes: E0 (EPL), SP1 (LaLiga), I1 (Serie A), D1 (Bundesliga), F1 (Ligue 1).
- Odds columns (B365H/B365D/B365A) may be missing for some seasons/leagues.
- This single file is intentionally self-contained. For a production repo, split
  into modules (ingest.py, features.py, train.py, app.py), add tests and CI.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

# -----------------------------
# Configuration & Utilities
# -----------------------------

LEAGUE_NAMES = {
    "E0": "Premier League",
    "SP1": "LaLiga",
    "I1": "Serie A",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
}

DEFAULT_LEAGUES = ["E0", "SP1", "I1", "D1", "F1"]

DATA_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
ARTIFACTS_DIR = os.path.join("artifacts")

os.makedirs(DATA_DIR, exist_ok=True)
(os.path.exists(PROCESSED_DIR) and True) or os.makedirs(PROCESSED_DIR)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


@dataclass
class TrainConfig:
    start_season: int = 2015  # e.g., 2015→2015-2016
    end_season: int = 2024    # inclusive, e.g., 2024→2024-2025
    leagues: List[str] = None
    test_last_seasons: int = 1
    validation_frac: float = 0.15  # fraction of pre-test data for validation/calibration
    random_state: int = 42
    elo_k: float = 20.0
    elo_home_adv: float = 55.0
    rolling_window: int = 5

    def __post_init__(self):
        if self.leagues is None:
            self.leagues = DEFAULT_LEAGUES


# -----------------------------
# Data Acquisition
# -----------------------------

FD_BASE = "https://www.football-data.co.uk/mmz4281"


def _season_code(year_start: int) -> str:
    """Football-data season code, e.g., 2015→"1516"; 2024→"2425"."""
    y1 = year_start % 100
    y2 = (year_start + 1) % 100
    return f"{y1:02d}{y2:02d}"


def _fd_url(season_start: int, league: str) -> str:
    sc = _season_code(season_start)
    return f"{FD_BASE}/{sc}/{league}.csv"


def download_league_season(season_start: int, league: str, out_dir: str = DATA_DIR) -> Optional[str]:
    """Download CSV for a given season & league. Returns local file path or None if failed."""
    url = _fd_url(season_start, league)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{league}_{season_start}_{season_start+1}.csv")

    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200 and len(resp.content) > 500:  # simple sanity check
            with open(out_path, "wb") as f:
                f.write(resp.content)
            return out_path
        else:
            print(f"[WARN] Failed/malformed download: {url} (status={resp.status_code})")
            return None
    except Exception as e:
        print(f"[ERROR] Download error for {url}: {e}")
        return None


def load_all_matches(cfg: TrainConfig) -> pd.DataFrame:
    """Download and load CSVs into a single DataFrame with unified schema."""
    paths = []
    for season in tqdm(range(cfg.start_season, cfg.end_season + 1), desc="Downloading CSVs"):
        for lg in cfg.leagues:
            p = download_league_season(season, lg)
            if p:
                paths.append((p, lg, season))

    frames = []
    for p, lg, season in paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            # Some files have Latin-1 encoding
            df = pd.read_csv(p, encoding="latin-1")

        # Standardize minimal needed columns; football-data schemas vary by year
        # We will keep: Div, Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, B365H, B365D, B365A, HS, AS, HST, AST
        # Missing columns will be added as NaN.
        needed = [
            "Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
            "B365H", "B365D", "B365A",
            "HS", "AS", "HST", "AST",
        ]
        for col in needed:
            if col not in df.columns:
                df[col] = np.nan

        # Parse date robustly (many files are dd/mm/yy or dd/mm/yyyy)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

        df = df[needed].copy()
        df["League"] = lg
        df["SeasonStart"] = season
        frames.append(df)

    if not frames:
        raise RuntimeError("No data downloaded. Check leagues/seasons or connectivity.")

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])  # remove rows with invalid dates/teams
    all_df.sort_values(["League", "Date"], inplace=True)
    all_df.reset_index(drop=True, inplace=True)
    return all_df


# -----------------------------
# Feature Engineering
# -----------------------------

def compute_elo_features(df: pd.DataFrame, k: float = 20.0, home_adv: float = 55.0) -> pd.DataFrame:
    """Compute pre-match Elo ratings per team and attach to each row.
    Returns df with columns: elo_home_pre, elo_away_pre, elo_diff
    """
    df = df.copy()
    df["elo_home_pre"] = np.nan
    df["elo_away_pre"] = np.nan

    # Maintain ratings per league
    for lg, g in df.groupby("League", sort=False):
        ratings: Dict[str, float] = {}
        for idx, row in g.sort_values("Date").iterrows():
            h, a = row["HomeTeam"], row["AwayTeam"]
            Rh = ratings.get(h, 1500.0)
            Ra = ratings.get(a, 1500.0)
            df.at[idx, "elo_home_pre"] = Rh
            df.at[idx, "elo_away_pre"] = Ra

            # After match, update Elo
            if pd.isna(row["FTR"]):
                continue
            # expected scores with home advantage
            Eh = 1.0 / (1.0 + 10 ** ( ( (Ra + home_adv) - Rh ) / 400.0 ))
            Ea = 1.0 / (1.0 + 10 ** ( ( (Rh - home_adv) - Ra ) / 400.0 ))

            if row["FTR"] == "H":
                Sh, Sa = 1.0, 0.0
            elif row["FTR"] == "A":
                Sh, Sa = 0.0, 1.0
            else:
                Sh, Sa = 0.5, 0.5

            Rh_new = Rh + k * (Sh - Eh)
            Ra_new = Ra + k * (Sa - Ea)
            ratings[h] = Rh_new
            ratings[a] = Ra_new

    df["elo_diff"] = df["elo_home_pre"] - df["elo_away_pre"]
    return df


def _result_points(is_home: bool, ftr: str) -> int:
    if ftr == "D":
        return 1
    if is_home and ftr == "H":
        return 3
    if (not is_home) and ftr == "A":
        return 3
    return 0


def build_team_long_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a long-form team match table with per-team rows for rolling features."""
    long_rows = []
    for idx, r in df.iterrows():
        # Home row
        long_rows.append({
            "match_idx": idx,
            "League": r["League"],
            "Date": r["Date"],
            "team": r["HomeTeam"],
            "opponent": r["AwayTeam"],
            "is_home": True,
            "goals_for": r.get("FTHG", np.nan),
            "goals_against": r.get("FTAG", np.nan),
            "shots": r.get("HS", np.nan),
            "shots_ot": r.get("HST", np.nan),
            "points": _result_points(True, r.get("FTR", np.nan)),
        })
        # Away row
        long_rows.append({
            "match_idx": idx,
            "League": r["League"],
            "Date": r["Date"],
            "team": r["AwayTeam"],
            "opponent": r["HomeTeam"],
            "is_home": False,
            "goals_for": r.get("FTAG", np.nan),
            "goals_against": r.get("FTHG", np.nan),
            "shots": r.get("AS", np.nan),
            "shots_ot": r.get("AST", np.nan),
            "points": _result_points(False, r.get("FTR", np.nan)),
        })
    long_df = pd.DataFrame(long_rows)
    long_df.sort_values(["League", "team", "Date"], inplace=True)
    long_df.reset_index(drop=True, inplace=True)
    return long_df


# def compute_rolling_form_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
#     """Compute pre-match rolling form features for each team and merge back to match-level."""
#     long_df = build_team_long_table(df)

#     # Shift by 1 so current match data isn't used in features
#     def _roll(group: pd.DataFrame) -> pd.DataFrame:
#         g = group.copy()
#         for col in ["goals_for", "goals_against", "shots", "shots_ot", "points"]:
#             g[f"{col}_roll_sum"] = (
#                 g[col].shift(1).rolling(window=window, min_periods=1).sum()
#             )
#             g[f"{col}_roll_mean"] = (
#                 g[col].shift(1).rolling(window=window, min_periods=1).mean()
#             )
#         # points per game last window
#         g["ppg_last_window"] = g["points"].shift(1).rolling(window=window, min_periods=1).mean()
#         return g

#     rolled = long_df.groupby(["League", "team"], group_keys=False).apply(_roll)

#     # Split into home/away features per match_idx and merge back
#     home_feats = rolled[rolled["is_home"]].copy()
#     away_feats = rolled[~rolled["is_home"]].copy()

#     home_feats = home_feats.add_prefix("home_")
#     away_feats = away_feats.add_prefix("away_")

#     # Keep only columns we care about
#     keep_cols = [
#         "match_idx",
#         "ppg_last_window",
#         "goals_for_roll_mean",
#         "goals_against_roll_mean",
#         "shots_roll_mean",
#         "shots_ot_roll_mean",
#     ]

#     home_keep = [f"home_{c}" for c in keep_cols]
#     away_keep = [f"away_{c}" for c in keep_cols]

#     home_feats = home_feats[home_keep]
#     away_feats = away_feats[away_keep]

#     merged = (
#         df.reset_index().rename(columns={"index": "match_idx"})
#         .merge(home_feats, on="match_idx", how="left")
#         .merge(away_feats, on="match_idx", how="left")
#         .drop(columns=["match_idx"])
#     )
#     return merged


def compute_rolling_form_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    long_df = build_team_long_table(df)

    def _roll(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        for col in ['goals_for','goals_against','shots','shots_ot','points']:
            g[f'{col}_roll_mean'] = g[col].shift(1).rolling(window=window, min_periods=1).mean()
        g['ppg_last_window'] = g['points'].shift(1).rolling(window=window, min_periods=1).mean()
        return g

    rolled = long_df.groupby(['League','team'], group_keys=False).apply(_roll)

    # Split home/away
    home = rolled[rolled['is_home']].copy()
    away = rolled[~rolled['is_home']].copy()

    keep = ['match_idx','ppg_last_window','goals_for_roll_mean','goals_against_roll_mean',
            'shots_roll_mean','shots_ot_roll_mean']

    # Rename only the feature columns (not match_idx)
    home = home[keep].rename(columns={
        'ppg_last_window':'home_ppg_last_window',
        'goals_for_roll_mean':'home_goals_for_roll_mean',
        'goals_against_roll_mean':'home_goals_against_roll_mean',
        'shots_roll_mean':'home_shots_roll_mean',
        'shots_ot_roll_mean':'home_shots_ot_roll_mean'
    })

    away = away[keep].rename(columns={
        'ppg_last_window':'away_ppg_last_window',
        'goals_for_roll_mean':'away_goals_for_roll_mean',
        'goals_against_roll_mean':'away_goals_against_roll_mean',
        'shots_roll_mean':'away_shots_roll_mean',
        'shots_ot_roll_mean':'away_shots_ot_roll_mean'
    })

    # Merge by the shared key
    base = df.reset_index().rename(columns={'index':'match_idx'})
    return (base
            .merge(home, on='match_idx', how='left')
            .merge(away, on='match_idx', how='left')
            .drop(columns=['match_idx']))


def add_implied_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert bookmaker odds to normalized implied probabilities if available."""
    df = df.copy()
    for cols in [("B365H", "B365D", "B365A")]:
        h, d, a = cols
        if all(c in df.columns for c in cols):
            inv_sum = 1.0/df[h] + 1.0/df[d] + 1.0/df[a]
            df["imp_home"] = (1.0/df[h]) / inv_sum
            df["imp_draw"] = (1.0/df[d]) / inv_sum
            df["imp_away"] = (1.0/df[a]) / inv_sum
        else:
            df["imp_home"] = np.nan
            df["imp_draw"] = np.nan
            df["imp_away"] = np.nan
    return df


def build_features(df: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    df1 = compute_elo_features(df, k=cfg.elo_k, home_adv=cfg.elo_home_adv)
    df2 = compute_rolling_form_features(df1, window=cfg.rolling_window)
    df3 = add_implied_odds_features(df2)

    # Target encoding
    df3 = df3.copy()
    df3 = df3.dropna(subset=["FTR"])  # only completed matches

    label_map = {"H": 0, "D": 1, "A": 2}
    df3["target"] = df3["FTR"].map(label_map)

    # Season label for time-based splits
    df3["SeasonStart"] = df3["SeasonStart"].astype(int)

    # Select model features
    feature_cols = [
        # Elo
        "elo_home_pre", "elo_away_pre", "elo_diff",
        # Rolling form (home)
        "home_ppg_last_window", "home_goals_for_roll_mean", "home_goals_against_roll_mean",
        "home_shots_roll_mean", "home_shots_ot_roll_mean",
        # Rolling form (away)
        "away_ppg_last_window", "away_goals_for_roll_mean", "away_goals_against_roll_mean",
        "away_shots_roll_mean", "away_shots_ot_roll_mean",
        # Odds
        "imp_home", "imp_draw", "imp_away",
    ]

    # Ensure all features exist
    for c in feature_cols:
        if c not in df3.columns:
            df3[c] = np.nan

    # Keep minimal columns
    keep = [
        "League", "Date", "HomeTeam", "AwayTeam", "SeasonStart", "target"
    ] + feature_cols

    final_df = df3[keep].sort_values(["League", "Date"]).reset_index(drop=True)
    return final_df


# -----------------------------
# Modeling
# -----------------------------

def time_based_split(df: pd.DataFrame, last_n_seasons_test: int = 1, val_frac: float = 0.15):
    """Split by season for train/val/test where test is the last N seasons.
    Validation is the last fraction of the pre-test data (time-ordered).
    """
    seasons = sorted(df["SeasonStart"].unique())
    test_seasons = seasons[-last_n_seasons_test:]

    df_trainval = df[~df["SeasonStart"].isin(test_seasons)].copy()
    df_test = df[df["SeasonStart"].isin(test_seasons)].copy()

    # Validation = last val_frac of trainval by date
    df_trainval = df_trainval.sort_values("Date")
    split_idx = int((1.0 - val_frac) * len(df_trainval))
    df_train = df_trainval.iloc[:split_idx]
    df_val = df_trainval.iloc[split_idx:]

    return df_train, df_val, df_test


def build_pipelines(random_state: int = 42, scale: bool = True):
    numeric_features = []  # will be set dynamically when fitting

    num_transformers = []
    num_transformers.append(("imputer", SimpleImputer(strategy="median")))
    if scale:
        num_transformers.append(("scaler", StandardScaler()))

    preprocessor = Pipeline(num_transformers)

    # Models
    logreg = LogisticRegression(max_iter=2000, multi_class="multinomial", n_jobs=None)
    hgb = HistGradientBoostingClassifier(random_state=random_state)

    pipe_logreg = Pipeline([
        ("prep", preprocessor),
        ("clf", logreg),
    ])
    pipe_hgb = Pipeline([
        ("prep", preprocessor),
        ("clf", hgb),
    ])

    return {"logreg": pipe_logreg, "hgb": pipe_hgb}


def evaluate_model(model: Pipeline, X: pd.DataFrame, y: pd.Series, label: str) -> Dict[str, float]:
    proba = model.predict_proba(X)
    preds = np.argmax(proba, axis=1)
    acc = accuracy_score(y, preds)
    ll = log_loss(y, proba, labels=[0, 1, 2])
    br = brier_score_loss(
        (y == 0).astype(int), proba[:, 0]
    )  # Brier on Home win (class 0) as a simple reference
    print(f"\n[{label}] Accuracy={acc:.3f}  LogLoss={ll:.3f}  Brier(Home)={br:.3f}")
    print("Confusion Matrix:\n", confusion_matrix(y, preds))
    print("Classification Report:\n", classification_report(y, preds, digits=3))
    return {"accuracy": acc, "log_loss": ll, "brier_home": br}


def fit_and_select(df: pd.DataFrame, feature_cols: List[str], cfg: TrainConfig):
    df_train, df_val, df_test = time_based_split(
        df, last_n_seasons_test=cfg.test_last_seasons, val_frac=cfg.validation_frac
    )

    X_train, y_train = df_train[feature_cols], df_train["target"]
    X_val, y_val = df_val[feature_cols], df_val["target"]
    X_test, y_test = df_test[feature_cols], df_test["target"]

    models = build_pipelines(cfg.random_state)

    metrics = {}
    best_key = None
    best_val = math.inf

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        print(f"\n=== {name} (Validation) ===")
        m = evaluate_model(pipe, X_val, y_val, label=f"{name}-val")
        metrics[name] = m
        if m["log_loss"] < best_val:
            best_val = m["log_loss"]
            best_key = name

    print(f"\nSelected base model: {best_key}")
    best_model = models[best_key]

    # Probability calibration on train+val using isotonic (time-aware approach would use a rolling scheme)
    calib = CalibratedClassifierCV(estimator=best_model, method="isotonic", cv=3)
    calib.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

    print("\n=== Calibrated Model (Test) ===")
    test_metrics = evaluate_model(calib, X_test, y_test, label="calibrated-test")

    # Permutation importance (on validation for speed)
    try:
        print("Computing permutation importance (validation)…")
        r = permutation_importance(calib, X_val, y_val, n_repeats=10, random_state=cfg.random_state)
        importances = (
            pd.DataFrame({"feature": feature_cols, "importance": r.importances_mean})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    except Exception as e:
        print(f"[WARN] Permutation importance failed: {e}")
        importances = pd.DataFrame({"feature": feature_cols, "importance": np.nan})

    return calib, metrics, test_metrics, importances, (df_train, df_val, df_test)


# -----------------------------
# Persistence & Prediction
# -----------------------------

def save_artifacts(model, feature_cols: List[str], cfg: TrainConfig, importances: pd.DataFrame):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    model_path = os.path.join(ARTIFACTS_DIR, "model.pkl")
    joblib.dump(model, model_path)

    with open(os.path.join(ARTIFACTS_DIR, "columns.json"), "w") as f:
        json.dump({"feature_cols": feature_cols}, f, indent=2)

    with open(os.path.join(ARTIFACTS_DIR, "metadata.json"), "w") as f:
        meta = {
            "train_config": asdict(cfg),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "league_names": LEAGUE_NAMES,
        }
        json.dump(meta, f, indent=2)

    importances.to_csv(os.path.join(ARTIFACTS_DIR, "feature_importance.csv"), index=False)
    print(f"Artifacts saved to: {ARTIFACTS_DIR}")


def _prepare_fixture_df(fixtures: List[Tuple[str, str, str]], history_df: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    """Build a feature frame for upcoming fixtures.
    fixtures: list of (date_str, home_team, away_team) with date ISO YYYY-MM-DD.
    We compute Elo & rolling features using historical df up to the given date.
    """
    # Use latest known ratings/forms up to date for each fixture
    # Strategy: For each fixture, find the latest match date per team before fixture date,
    # then borrow that team's last known rolling features and Elo.
    # If team not found, default to neutral values.

    # Build helper tables from processed df (which already contains pre-match features)
    hist = history_df.copy()
    hist.sort_values("Date", inplace=True)

    # Use the *pre-match* features corresponding to each team's last played match
    # Create per-team snapshots for home & away feature sets
    team_home_cols = [c for c in hist.columns if c.startswith("home_")]
    team_away_cols = [c for c in hist.columns if c.startswith("away_")]

    # We'll store last known (home_*) features when team played at home, similarly for away
    # For Elo, we can use elo_home_pre/elo_away_pre depending on role; use their mean as a proxy
    snapshots = {}
    for team in pd.unique(pd.concat([hist["HomeTeam"], hist["AwayTeam"]])):
        team_hist = hist[(hist["HomeTeam"] == team) | (hist["AwayTeam"] == team)]
        if team_hist.empty:
            continue
        # Last row for the team
        last_row = team_hist.iloc[-1]
        # Approximate team's current Elo as the average of home/away pre in last appearance
        elo_est = np.nanmean([last_row.get("elo_home_pre"), last_row.get("elo_away_pre")])
        # For rolling, take whichever role they last played; then map to neutral names
        if last_row["HomeTeam"] == team:
            ppg = last_row.get("home_ppg_last_window", np.nan)
            gf = last_row.get("home_goals_for_roll_mean", np.nan)
            ga = last_row.get("home_goals_against_roll_mean", np.nan)
            shots = last_row.get("home_shots_roll_mean", np.nan)
            sOT = last_row.get("home_shots_ot_roll_mean", np.nan)
        else:
            ppg = last_row.get("away_ppg_last_window", np.nan)
            gf = last_row.get("away_goals_for_roll_mean", np.nan)
            ga = last_row.get("away_goals_against_roll_mean", np.nan)
            shots = last_row.get("away_shots_roll_mean", np.nan)
            sOT = last_row.get("away_shots_ot_roll_mean", np.nan)

        snapshots[team] = {
            "ppg": ppg,
            "gf": gf,
            "ga": ga,
            "shots": shots,
            "shots_ot": sOT,
            "elo": elo_est,
        }

    # Build fixture feature rows
    rows = []
    for date_str, home, away in fixtures:
        d = pd.to_datetime(date_str)
        h = snapshots.get(home, {"ppg": np.nan, "gf": np.nan, "ga": np.nan, "shots": np.nan, "shots_ot": np.nan, "elo": 1500.0})
        a = snapshots.get(away, {"ppg": np.nan, "gf": np.nan, "ga": np.nan, "shots": np.nan, "shots_ot": np.nan, "elo": 1500.0})
        row = {
            "League": "E0",  # unknown; only used for display
            "Date": d,
            "HomeTeam": home,
            "AwayTeam": away,
            "SeasonStart": d.year if d.month >= 7 else d.year - 1,
            # Elo
            "elo_home_pre": h["elo"],
            "elo_away_pre": a["elo"],
            "elo_diff": h["elo"] - a["elo"],
            # Rolling (use snapshots)
            "home_ppg_last_window": h["ppg"],
            "home_goals_for_roll_mean": h["gf"],
            "home_goals_against_roll_mean": h["ga"],
            "home_shots_roll_mean": h["shots"],
            "home_shots_ot_roll_mean": h["shots_ot"],
            "away_ppg_last_window": a["ppg"],
            "away_goals_for_roll_mean": a["gf"],
            "away_goals_against_roll_mean": a["ga"],
            "away_shots_roll_mean": a["shots"],
            "away_shots_ot_roll_mean": a["shots_ot"],
            # Odds unknown for future fixtures
            "imp_home": np.nan,
            "imp_draw": np.nan,
            "imp_away": np.nan,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def predict_fixtures(fixtures: List[Tuple[str, str, str]], model_path: str = os.path.join(ARTIFACTS_DIR, "model.pkl"), cols_path: str = os.path.join(ARTIFACTS_DIR, "columns.json"), history_path: str = os.path.join(PROCESSED_DIR, "matches.parquet")) -> pd.DataFrame:
    model = joblib.load(model_path)
    with open(cols_path, "r") as f:
        feature_cols = json.load(f)["feature_cols"]

    hist = pd.read_parquet(history_path)
    feat_df = _prepare_fixture_df(fixtures, hist, cfg=TrainConfig())
    X = feat_df[feature_cols]
    proba = model.predict_proba(X)

    # Map class index back to label
    inv_map = {0: "H", 1: "D", 2: "A"}
    preds = proba.argmax(axis=1)

    out = feat_df[["Date", "HomeTeam", "AwayTeam"]].copy()
    out["pred_H"] = proba[:, 0]
    out["pred_D"] = proba[:, 1]
    out["pred_A"] = proba[:, 2]
    out["pred_label"] = [inv_map[i] for i in preds]
    return out


# -----------------------------
# Orchestration
# -----------------------------

def run_training(cfg: TrainConfig):
    print("Loading matches…")
    raw_df = load_all_matches(cfg)
    print(f"Loaded {len(raw_df):,} rows")

    print("Building features…")
    feat_df = build_features(raw_df, cfg)
    print(f"Feature rows: {len(feat_df):,}")

    # Persist processed dataset for reuse
    out_parquet = os.path.join(PROCESSED_DIR, "matches.parquet")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    feat_df.to_parquet(out_parquet, index=False)
    print(f"Saved processed dataset → {out_parquet}")

    feature_cols = [c for c in feat_df.columns if c not in {"League", "Date", "HomeTeam", "AwayTeam", "SeasonStart", "target"}]

    print("Training & selecting model…")
    model, val_metrics, test_metrics, importances, splits = fit_and_select(feat_df, feature_cols, cfg)

    print("Saving artifacts…")
    save_artifacts(model, feature_cols, cfg, importances)

    print("Done.")


def parse_fixtures_list(fixtures_list: List[str]) -> List[Tuple[str, str, str]]:
    out = []
    for item in fixtures_list:
        parts = [p.strip() for p in item.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Fixture must be 'YYYY-MM-DD,HomeTeam,AwayTeam': {item}")
        out.append((parts[0], parts[1], parts[2]))
    return out


def build_arg_parser():
    p = argparse.ArgumentParser(description="Soccer Match Predictor Pipeline")
    p.add_argument("--start_season", type=int, default=2015, help="First season start year (e.g., 2015 for 2015-2016)")
    p.add_argument("--end_season", type=int, default=2024, help="Last season start year (inclusive)")
    p.add_argument("--leagues", nargs="*", default=DEFAULT_LEAGUES, help="League codes, e.g., E0 SP1 I1 D1 F1")
    p.add_argument("--test_last_seasons", type=int, default=1)
    p.add_argument("--validation_frac", type=float, default=0.15)
    p.add_argument("--elo_k", type=float, default=20.0)
    p.add_argument("--elo_home_adv", type=float, default=55.0)
    p.add_argument("--rolling_window", type=int, default=5)
    p.add_argument("--predict", action="store_true", help="If set, skip training and predict provided fixtures")
    p.add_argument("--fixtures", nargs="*", help="Fixtures as 'YYYY-MM-DD,HomeTeam,AwayTeam'")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    cfg = TrainConfig(
        start_season=args.start_season,
        end_season=args.end_season,
        leagues=args.leagues,
        test_last_seasons=args.test_last_seasons,
        validation_frac=args.validation_frac,
        elo_k=args.elo_k,
        elo_home_adv=args.elo_home_adv,
        rolling_window=args.rolling_window,
    )

    if args.predict:
        if not args.fixtures:
            raise SystemExit("--predict requires --fixtures 'YYYY-MM-DD,HomeTeam,AwayTeam' …")
        fixtures = parse_fixtures_list(args.fixtures)
        preds = predict_fixtures(fixtures)
        print(preds.to_string(index=False))
    else:
        run_training(cfg)
