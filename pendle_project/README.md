# 🌀 Pendle Project

Pendle is a simulation and modeling system designed to explore and forecast rare economic stress events. The project combines public datasets, synthetic stress environments, PCA-based latent abstractions, and neural extrapolation (CNNs), while optionally integrating non-economic signals like sentiment and media trends.

---

## 🔭 Mission

To simulate, predict, and interpret economic shocks using a modular pipeline that blends:
- Real-world macro & financial signals (FRED, Yahoo, Trends)
- Synthetic rare-event data with stressor variations (sparsity, drift, noise, etc.)
- PCA scalarization for latent generalization
- Model ensemble tournaments (ElasticNet, SGD, XGBoost, CNN)
- Extrapolated forecasting using shape-aware CNNs
- Integration of non-economic layers (pop culture, elections, media fear index)

---

## 🧠 Week-by-Week Flow

### **Week 1 — Public Data Prediction**
- Ingest freely available data (FRED, Yahoo Finance, Google Trends)
- Impute, clean, normalize
- Build baseline economic stress predictor
- Notebook: `week01_EDA_public_data.ipynb`

---

### **Week 2 — Synthetic Rare Event Ensembles**
- Generate diverse synthetic stress environments:
  - Control for noise, sparsity, drift, imbalance
- Train baseline rare-event classifiers
- Evaluate with tournament logic
- Notebook: `week02_synthetic_baseline_ensemble.ipynb`

---

### **Week 3 — Latent Feature Templates via PCA**
- Scalarize inputs and build PCA abstraction layer
- Map back to interpretable labels
- Unblind the latent space post-modeling
- Notebook: `week03_pca_blinding_unblinding.ipynb`

---

### **Week 4 — Evaluation on Real Data**
- Blend real economic + alternative signals
- Test champion models trained on synthetic data
- Compare lift, PR AUC, extrapolation ability
- Notebook: `week04_eval_real_data_models.ipynb`

---

### **Week 5a — CNN Extrapolation Forecaster**
- Build CNN on PCA trajectories
- Predict plausible forward latent states
- Convert back to real-world ranges
- Notebook: `week05a_cnn_forecasting_extrapolation.ipynb`

---

### **Week 5b — Non-Economic Signals Integration**
- Ingest sentiment, Reddit, TikTok, GPT weights
- Merge with macro signals
- Test improvement in early stress signal detection
- Notebook: `week05b_popculture_signals_layer.ipynb`

---

## 🗂 Folder Structure

```
pendle_project/
├── data/              # Raw, synthetic, processed datasets
├── src/               # Modular code: ingestion, cleaning, modeling, forecasting
├── notebooks/         # Week-by-week Jupyter experiments
├── models/            # Trained models and CNNs
├── outputs/           # Logs, visualizations, extrapolated forecasts
├── docs/              # Slides, theory notes, reference materials
├── demo/              # Optional Streamlit app (WIP)
├── tests/             # Unit tests for components
├── run_demo.py        # CLI launcher for the demo
└── .gitignore         # Git exclusions
```

---

## 🗂 Comprehensive Folder and File Structure

```
pendle_project/
📁 pendle_project/
    📄 .gitignore
    📄 README.md
    📄 run_demo.py

    📁 data/
        📁 raw/
            📁 fred/
            📁 yfinance/
            📁 trends/
            📁 alt_signals/
        📁 synthetic/
            📁 easy/
            📁 medium/
            📁 hard/
            📁 extrapolated/
        📁 processed/
        📁 mappings/
        📁 sim_config/

    📁 src/
        📁 ingestion/
            📄 __init__.py
        📁 cleaning/
            📄 impute.py
        📁 synthesis/
            📄 generator.py               # ✅ Updated to support extended rare event parameters
        📁 pca_transform/
            📄 __init__.py
        📁 extrapolation/
            📄 cnn_forecaster.py
        📁 modeling/
            📄 __init__.py
        📁 evaluation/
            📄 __init__.py
        📁 utils/
            📄 __init__.py

    📁 notebooks/
        📄 week01_EDA_public_data.ipynb
        📄 week02_synthetic_baseline_ensemble.ipynb
        📄 week03_pca_blinding_unblinding.ipynb
        📄 week04_eval_real_data_models.ipynb
        📄 week05a_cnn_forecasting_extrapolation.ipynb
        📄 week05b_popculture_signals_layer.ipynb
        📁 exploratory/
            📄 README.md
        📁 pipelines/
            📄 README.md

    📁 models/
        📁 baselines/
        📁 champions/
        📁 cnn_forecasters/
        📁 metadata/

    📁 outputs/
        📁 figures/
            📁 latent_shapes/
            📁 market_forecast_trends/
            📁 correlation_maps/
        📁 logs/
            📁 training/
            📁 breeding_battles/
        📁 leaderboard/
        📁 forecasts/

    📁 tests/

    📁 docs/
        📁 theory/

    📁 demo/
        📄 streamlit_app.py              # ✅ NEW: Main UI for interactive simulation + prediction
        📁 components/
            📄 forecast_demo.py          # ✅ NEW: Forecasting logic from keywords
            📄 keyword_mapping.py        # ✅ NEW: Mapping of keywords to simulation configs
```

---

## 🛠 Tools

- Python 3.11
- Pandas, Scikit-learn, XGBoost
- TensorFlow/Keras (CNNs)
- Matplotlib / Plotly for visuals

---

## 🧪 Sim Dimensions Supported

- Sparsity (missing data)
- Drift (gradual change)
- Noise (Gaussian, outlier)
- Imbalance (rare classes)
- Cyclicality / Seasonality
- Regime shifts / Black swans
- Label ambiguity & confounding signals

---

## ✅ Getting Started

```bash
# Install dependencies
conda env create -f environment.yml
conda activate pendle

# Launch a demo
python run_demo.py
```

---

## 🧬 Authors
Built by Sophia Boettcher and friends. Inspired by biology, finance, chaos, and code.
