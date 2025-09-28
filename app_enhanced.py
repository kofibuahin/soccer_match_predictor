
import json
import os
from dataclasses import dataclass
from datetime import date
from io import StringIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import joblib

from sklearn.metrics import brier_score_loss, log_loss
from sklearn.inspection import permutation_importance
import plotly.express as px

# ----------------------
# Config & Utilities
# ----------------------
@dataclass
class AppPaths:
    artifacts_dir: str = "artifacts"
    processed_dir: str = os.path.join("data", "processed")

@st.cache_resource
def load_artifacts(paths: AppPaths):
    model_path = os.path.join(paths.artifacts_dir, "model.pkl")
    cols_path = os.path.join(paths.artifacts_dir, "columns.json")
    meta_path = os.path.join(paths.artifacts_dir, "metadata.json")
    ds_path = os.path.join(paths.processed_dir, "matches.parquet")
    fi_path = os.path.join(paths.artifacts_dir, "feature_importance.csv")

    model = joblib.load(model_path)
    with open(cols_path, "r") as f:
        feature_cols = json.load(f)["feature_cols"]
    with open(meta_path, "r") as f:
        meta = json.load(f)
    hist = pd.read_parquet(ds_path)
    teams = sorted(pd.unique(pd.concat([hist["HomeTeam"], hist["AwayTeam"]])))

    fi = None
    if os.path.exists(fi_path):
        try:
            fi = pd.read_csv(fi_path)
        except Exception:
            fi = None

    return model, feature_cols, meta, hist, teams, fi

def prepare_fixture_df(fixtures: List[Tuple[str, str, str]], history_df: pd.DataFrame) -> pd.DataFrame:
    # Build minimal feature rows for upcoming fixtures using last-known team snapshots from history_df.
    hist = history_df.copy().sort_values("Date")
    snapshots: Dict[str, Dict[str, float]] = {}

    all_teams = pd.unique(pd.concat([hist["HomeTeam"], hist["AwayTeam"]]))
    for team in all_teams:
        th = hist[(hist["HomeTeam"] == team) | (hist["AwayTeam"] == team)]
        if th.empty:
            continue
        last = th.iloc[-1]
        elo_est = np.nanmean([last.get("elo_home_pre"), last.get("elo_away_pre")])
        if last["HomeTeam"] == team:
            ppg = last.get("home_ppg_last_window"); gf = last.get("home_goals_for_roll_mean")
            ga = last.get("home_goals_against_roll_mean"); sh = last.get("home_shots_roll_mean")
            sot = last.get("home_shots_ot_roll_mean")
        else:
            ppg = last.get("away_ppg_last_window"); gf = last.get("away_goals_for_roll_mean")
            ga = last.get("away_goals_against_roll_mean"); sh = last.get("away_shots_roll_mean")
            sot = last.get("away_shots_ot_roll_mean")
        snapshots[team] = {
            "ppg": ppg, "gf": gf, "ga": ga,
            "shots": sh, "shots_ot": sot,
            "elo": float(elo_est) if not np.isnan(elo_est) else 1500.0
        }
        
    rows = []
    for date_str, home, away in fixtures:
        d = pd.to_datetime(date_str)
        h = snapshots.get(home, {"ppg": np.nan, "gf": np.nan, "ga": np.nan, "shots": np.nan, "shots_ot": np.nan, "elo": 1500.0})
        a = snapshots.get(away, {"ppg": np.nan, "gf": np.nan, "ga": np.nan, "shots": np.nan, "shots_ot": np.nan, "elo": 1500.0})
        rows.append({
            "League": "E0",
            "Date": d,
            "HomeTeam": home,
            "AwayTeam": away,
            "SeasonStart": d.year if d.month >= 7 else d.year - 1,
            "elo_home_pre": h["elo"],
            "elo_away_pre": a["elo"],
            "elo_diff": h["elo"] - a["elo"],
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
            "imp_home": np.nan, "imp_draw": np.nan, "imp_away": np.nan,
        })
    return pd.DataFrame(rows)

def ev_table_from_odds(proba: np.ndarray, odds: Tuple[float, float, float]):
    # Compute expected value given decimal odds (home, draw, away).
    outcomes = np.array(["Home", "Draw", "Away"])
    payouts = np.array(odds) - 1.0  # net profit per $1
    ev = proba * payouts - (1 - proba)
    return pd.DataFrame({"Outcome": outcomes, "Model_Prob": proba, "Odds": odds, "EV_per_$1": ev})

def reliability_curve(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10):
    # Binary reliability curve for class 'Home' (extend per-class if desired).
    p = proba[:, 0]
    df = pd.DataFrame({"y": (y_true == 0).astype(int), "p": p})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    agg = df.groupby("bin").agg(emp_rate=("y", "mean"), avg_p=("p", "mean"), count=("y", "size")).reset_index()
    return agg

# ----------------------
# UI
# ----------------------
st.set_page_config(page_title="Soccer Match Predictor", page_icon="⚽", layout="wide")
st.title("⚽ Soccer Match Predictor")

with st.sidebar:
    st.header("Paths")
    artifacts_dir = st.text_input("Artifacts directory", value="artifacts")
    processed_dir = st.text_input("Processed data dir", value=os.path.join("data", "processed"))
    paths = AppPaths(artifacts_dir=artifacts_dir, processed_dir=processed_dir)
    load_btn = st.button("Load model & data", type="primary")

if "ready" not in st.session_state:
    st.session_state.ready = False

if load_btn:
    try:
        model, feature_cols, meta, hist, teams, fi = load_artifacts(paths)
        st.session_state.update({
            "ready": True,
            "model": model,
            "feature_cols": feature_cols,
            "meta": meta,
            "hist": hist,
            "teams": teams,
            "fi": fi
        })
        st.success("Artifacts loaded.")
    except Exception as e:
        st.session_state.ready = False
        st.error(f"Failed to load artifacts: {e}")

if not st.session_state.ready:
    st.info("Use the sidebar to load your saved artifacts (after training).")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Single Prediction", "Batch Predictions", "Calibration", "Feature Importance"])

with tab1:
    st.subheader("Single Match Prediction")
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        match_date = st.date_input("Match date", value=date.today())
    with col2:
        home_team = st.selectbox("Home team", st.session_state.teams, index=0)
    with col3:
        away_team = st.selectbox("Away team", st.session_state.teams, index=1)

    if home_team == away_team:
        st.warning("Home and away teams must be different.")
    else:
        odds_col1, odds_col2, odds_col3 = st.columns(3)
        with odds_col1:
            odd_h = st.number_input("Decimal odds (Home)", value=0.0, min_value=0.0, step=0.01, help="Optional")
        with odds_col2:
            odd_d = st.number_input("Decimal odds (Draw)", value=0.0, min_value=0.0, step=0.01, help="Optional")
        with odds_col3:
            odd_a = st.number_input("Decimal odds (Away)", value=0.0, min_value=0.0, step=0.01, help="Optional")

        if st.button("Predict", type="primary"):
            fx = prepare_fixture_df([(match_date.strftime("%Y-%m-%d"), home_team, away_team)], st.session_state.hist)
            X = fx[st.session_state.feature_cols]
            proba = st.session_state.model.predict_proba(X)[0]
            classes = np.array(["Home", "Draw", "Away"])
            pred_idx = int(np.argmax(proba))

            st.markdown(f"### Prediction: **{classes[pred_idx]}**")
            pred_df = pd.DataFrame({"Outcome": classes, "Probability": np.round(proba, 4)})
            st.dataframe(pred_df, use_container_width=True)
            st.plotly_chart(px.bar(pred_df, x="Outcome", y="Probability", title="Predicted Probabilities"), use_container_width=True)

            if odd_h > 0 and odd_d > 0 and odd_a > 0:
                ev_df = ev_table_from_odds(proba, (odd_h, odd_d, odd_a))
                st.markdown("#### Expected Value vs. Your Odds (per $1 stake)")
                st.dataframe(ev_df, use_container_width=True)
                st.plotly_chart(px.bar(ev_df, x="Outcome", y="EV_per_$1", title="EV by Outcome"), use_container_width=True)

with tab2:
    st.subheader("Batch Predictions")
    st.write("Upload a CSV with columns: `date,home_team,away_team` **or** paste rows below (comma-separated).")

    uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)
    text_rows = st.text_area("Or paste rows", placeholder="2025-08-15,Manchester City,Arsenal\n2025-08-16,Barcelona,Real Madrid")

    batch_df = None
    if uploaded is not None:
        try:
            tmp = pd.read_csv(uploaded)
            batch_df = tmp.rename(columns={
                "date":"date", "home_team":"home_team", "away_team":"away_team"
            })[["date","home_team","away_team"]]
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
    elif text_rows.strip():
        try:
            s = StringIO(text_rows.strip())
            tmp = pd.read_csv(s, header=None, names=["date","home_team","away_team"])
            batch_df = tmp
        except Exception as e:
            st.error(f"Failed to parse pasted rows: {e}")

    if batch_df is not None and not batch_df.empty:
        fixtures = [(str(r["date"]), str(r["home_team"]), str(r["away_team"])) for _, r in batch_df.iterrows()]
        fx = prepare_fixture_df(fixtures, st.session_state.hist)
        X = fx[st.session_state.feature_cols]
        proba = st.session_state.model.predict_proba(X)
        out = fx[["Date","HomeTeam","AwayTeam"]].copy()
        out["pred_H"] = proba[:,0]
        out["pred_D"] = proba[:,1]
        out["pred_A"] = proba[:,2]
        out["pred_label"] = np.array(["H","D","A"])[proba.argmax(axis=1)]
        st.dataframe(out, use_container_width=True)

        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions CSV", data=csv, file_name="batch_predictions.csv", mime="text/csv")

with tab3:
    st.subheader("Calibration / Reliability (Home class)")
    st.caption("Computed on historical matches using the trained model's predictions for the Home outcome (class 0).")

    hist = st.session_state.hist.dropna(subset=["target"]).copy()
    X_hist = hist[st.session_state.feature_cols]
    y_hist = hist["target"].astype(int).values
    proba_hist = st.session_state.model.predict_proba(X_hist)

    rel = reliability_curve(y_hist, proba_hist, n_bins=10)
    st.plotly_chart(px.scatter(rel, x="avg_p", y="emp_rate", size="count", title="Home Outcome Reliability: predicted vs empirical"), use_container_width=True)
    st.write("**Ideal line** is y=x. Deviations indicate miscalibration.")

    home_brier = brier_score_loss((y_hist==0).astype(int), proba_hist[:,0])
    mc_logloss = log_loss(y_hist, proba_hist, labels=[0,1,2])
    st.write(f"Brier (Home): **{home_brier:.3f}**   |   Multiclass LogLoss: **{mc_logloss:.3f}**")

with tab4:
    st.subheader("Feature Importance")
    fi = st.session_state.fi
    if fi is not None and {"feature","importance"}.issubset(fi.columns):
        st.write("Loaded from artifacts/feature_importance.csv")
        st.dataframe(fi.sort_values("importance", ascending=False), use_container_width=True)
        st.plotly_chart(px.bar(fi.sort_values("importance", ascending=False).head(20), x="feature", y="importance", title="Permutation Importance (Top 20)"), use_container_width=True)
    else:
        st.write("No saved feature importance found. Computing a lightweight permutation importance on a sample (up to 1500 rows).")
        sample = st.session_state.hist.dropna(subset=["target"]).sample(n=min(1500, len(st.session_state.hist)), random_state=42)
        Xs = sample[st.session_state.feature_cols]
        ys = sample["target"].astype(int).values
        try:
            r = permutation_importance(st.session_state.model, Xs, ys, n_repeats=5, random_state=42)
            fi2 = pd.DataFrame({"feature": st.session_state.feature_cols, "importance": r.importances_mean}).sort_values("importance", ascending=False)
            st.dataframe(fi2, use_container_width=True)
            st.plotly_chart(px.bar(fi2.head(20), x="feature", y="importance", title="Permutation Importance (Top 20)"), use_container_width=True)
        except Exception as e:
            st.error(f"Could not compute permutation importance: {e}")
