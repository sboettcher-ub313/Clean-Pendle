
# 🧠 A1_CreatingDatasets: Financial & Market Signal Aggregation

This module engineers datasets that explore relationships between macroeconomic trends, market sentiment, and rare financial stress events. It blends traditional finance, public search interest, and AI-inspired anomaly detection to create structured inputs for downstream forecasting and simulation.

---

## 📦 Project Purpose

We aim to build a time-aware feature matrix that detects instability in the financial system, using:
- **Yahoo Finance**, **FRED**, and **Google Trends**
- Z-scored, lagged, and rolling signals
- Labeled stress periods for rare-event training
- Media overlays to track social and emotional undercurrents

---

## 🧱 Directory Structure (Annotated)

```
A1_CreatingDatasets/
├── config/                  # Modeling and architecture config (NEAT, population, etc.)
│   ├── neat_config.txt
│   └── neat_config2.txt
│
├── data/
│   ├── raw/                # Placeholder for original data dumps (Yahoo, FRED)
│   ├── external/           # Non-core datasets (e.g., India mutual funds)
│   │   └── motilaloswalmfi.csv
│   ├── logs/               # Training run logs for model tracking/debugging
│   │   ├── output_log[1-4].txt
│   ├── processed/          # Core datasets powering analysis and modeling
│   │   ├── combined_df.csv
│   │   ├── financial_data_cleaned*.csv
│   │   ├── financial_data_btc_era.csv
│   │   ├── india_mutual_funds_cleaned.csv
│   │   └── market_stress.csv
│
├── exports/                # Longform writeups, HTML/PDF snapshots of reports
│   ├── March 23.html
│   └── The WoOWOo Side of Economics.{pdf,mhtml}
│
├── images/
│   ├── outputs/            # Visualizations generated from notebooks
│   │   ├── zscores.png, market_stress_top_features.png
│   │   ├── cnn.png, neat.png, diffusion.png, elasticnet.jpeg, galr.png
│   └── references/         # Meme, sentiment, news overlays (qualitative context)
│       └── *.png, *.gif, *.webp, *.jpg
│
├── notebooks/              # Analysis and generation notebooks
│   ├── DataAgg.ipynb       # Master data aggregation pipeline
│   ├── A2.ipynb            # Lagged indicators and feature logic
│   ├── A3.ipynb            # Tagging stress events and z-score anomalies
│   ├── bioinspo_financialmodeling.ipynb
│   └── Spellbook[2].ipynb, March 23.ipynb, Untitled.ipynb
│
└── scripts/                # Optional Python utilities for CLI or modularization
```

---

## 🧭 Execution Guide

1. Run `DataAgg.ipynb` to collect and merge raw datasets → `combined_df.csv`
2. Continue with `A2.ipynb` for rolling stats, temporal features
3. Use `A3.ipynb` to identify market stress periods (volatility + drawdown tagging)
4. Optional: View narratives in `March 23.ipynb` or `bioinspo_financialmodeling.ipynb`

Configs live in `/config/`, and logs are dumped to `/data/logs/`

---

## 🧠 Methods Summary

- **Rolling window smoothing** for volatility and price trends
- **Z-score normalization** across heterogeneous metrics
- **Crash tagging** using threshold breaks and trend direction
- **Sentiment injection** via Google Trends & Reddit proxy signals

---

## 🖼️ Key Visual Outputs

- Z-score anomalies over time (`zscores.png`)
- Top contributors to market stress (`market_stress_top_features.png`)
- Model performance heatmaps (`cnn_models.jpeg`, `diffusion.png`, `neat.png`)
- Meme economy references and visual overlays (`buy_sell.webp`, `fear_greed_1.png`, etc.)

---

## 🔁 How to Reproduce

- Python 3.9+ (with Jupyter, Pandas, Numpy, Matplotlib)
- External sources: Yahoo Finance, FRED, Google Trends (manually pulled or cached)
- Required: NEAT config if using evolutionary models

---

## ✨ Narrative Add-Ons

- `March 23.ipynb` and `The WoOWOo Side of Economics.pdf` explore interpretive storytelling layered on data
- Visual storytelling includes fear/greed, climate/occupancy imagery, cultural overlays

---

## 🧱 Future Ideas

- Modularize with `scripts/load_data.py`, `plot_utils.py`
- Integrate more live sources via APIs
- Create `data_dictionary.md` and `pipeline_overview.md` for onboarding

---

## 🧾 Credit

Curated and structured by Sophia Boettcher, 2025  
Design language inspired by diagnostic medicine, anomaly detection in ML, and storytelling in economic journalism.
