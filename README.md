# ⚽ Soccer Match Predictor

A machine learning project that predicts soccer match outcomes (Home / Draw / Away) using historical match data, team form, Elo ratings, and optional betting odds.  
Built with **Python, scikit-learn, pandas, and Streamlit**.

You can find a link to the live app here: https://soccermatchpredictor-7tbkfgtqzauj8txikefkdv.streamlit.app/

---

## 🚀 Features
- Train and evaluate ML models on historical soccer data
- Streamlit app with:
  - **Single match prediction** (with optional odds → EV calculation)
  - **Batch predictions** (CSV upload or paste fixtures)
  - **Calibration curve** (check probability reliability)
  - **Feature importance** (precomputed or live via permutation importance)
- Save and reuse artifacts (`model.pkl`, `columns.json`, `metadata.json`, etc.)
- Public deployment ready (Streamlit Community Cloud / Hugging Face Spaces / Docker)

---

## 🗂 Project Structure
