
# 🧠 A3_FeatureEngineering: Rare Event Simulation & Feature Layer Construction

This module powers the core **feature simulation, engineering, and model evolution** system used for rare event detection in synthetic financial scenarios. It unifies modular data generation, stacked modeling, champion selection, and diagnostic visualization.

---

## 📌 Objective

To generate robust training data representing rare-but-impactful financial "shock" scenarios, and evaluate multiple model families (e.g. GA-optimized LR, CNN, XGBoost) using evolutionary strategies. The best-performing models are promoted, stacked, and tracked over time for statistical performance and interpretability.

---

## 🧭 Workflow Summary

1. **Generate synthetic datasets** using realistic economic shock profiles (imbalance, drift, sparse noise).
2. **Run evolution tournaments** between model types using `breed_and_battle.py`.
3. **Save winners** to `models/` with metadata and predictions.
4. **Track results** across generations using visualizations and exports.
5. **Aggregate champions** into XGBoost or ensemble stacks.

---

## 📂 Folder Structure

```
A3_FeatureEngineering/
│
├── config/                         # (Placeholder for future configs or model settings)
│
├── images/
│   ├── references/                # Supporting images (1: image.png)
│   └── outputs/                   # PR AUC generation plots, diagnostic visuals
│       ├── Distribution of PR AUC by Generation Bins.png
│       ├── Model Performance over Generations (PR AUC).png
│       └── Projected Average PR AUC up to 900 Generations.png
│
├── models/
│   ├── trial_run/                 # First-run winners by simulation type
│   ├── champion_packages/        # Formal tournament winners with meta
│   ├── xgboost_stacks/           # XGB ensemble by scenario
│   └── champion_stacks/          # Final stacked models (non-XGB)
│
├── exports/
│   ├── Economic_Feature_Taxonomy.pdf         # Conceptual signal categories
│   ├── RareEvent_BreedAndBattle_WithInsights.pdf
│   ├── ShockSynth_WithLevels.pdf
│   ├── Step1_Initial_Exploration.pdf
│   ├── Step2_Liftoff_Baseline.pdf
│   ├── Step3_Lift_Booster.pdf
│   ├── Step4_Lift_Booster.pdf
│   ├── Step5_CausalDiagnostics.pdf
│   ├── Stockpocalypse.pdf
│   ├── Unblind_Features_PCA.pdf
│   ├── training_data_generator.pdf
│   └── gladatorial_viz/
│       ├── battle_arena.html
│       └── battle_log_data.js
│
├── scripts/
│   ├── SyntheticDataGenerator.py             # Builds custom drift/sparse/imbalanced sets
│   ├── MarketSimGenerators.py                # Economic signal types and PCA mixing
│   ├── breed_and_battle.py                   # Model evolution logic
│   ├── core_models.py, base_models.py        # Classifiers (XGB, ElasticNet, CNN, etc.)
│   ├── battle_logger.py, battle_comments.py  # Logging and commentary utilities
│   └── arena_upgrade_cell.py                 # Jupyter display logic (battle viz)
│
└── README.md                                 # This file
```

---

## 📊 Feature Engineering Highlights

- **Shock types** include:
  - `baseline_easy`, `high_drift`, `imbalanced_sparse`, `mixed_realistic`, `noisy_overlap`
- **Generation scoring** tracks PR AUC evolution over time
- **Meta files** store information on each model's config and context

---

## 🧠 Modeling Strategy

- Uses custom synthetic generators per economic theory or edge case
- Models "battle" over generations — survivors are saved as `.pkl`
- PR AUC tracked and visualized across time (up to 900 generations)
- Stacking for strong ensemble generalization (`champion_stacks/`, `xgboost_stacks/`)

---

## 📈 Outputs & Diagnostics

- Generation vs. PR AUC curves
- Top-performing models per scenario (by filename convention)
- Stacked champions by simulation class (easy/medium/hard/extreme)
- Final insight PDFs for sharing or publication

---

## 📤 How to Use

```bash
# Generate synthetic datasets
python scripts/SyntheticDataGenerator.py

# Run evolution and training
python scripts/breed_and_battle.py

# Analyze results
View outputs in images/outputs/ and exports/*.pdf
```

---

## 📁 Replaces the following previous READMEs

- `README arena.md`
- `README liftoff.md`
- `README simulated data.md`
- `README unblinding step.md`

Their contents have been consolidated and updated here for clarity and maintainability.

---

## 👩‍🔬 Maintainer

Developed and structured by Sophia Boettcher, 2025  
Inspired by tournament modeling, bioinspired learning, and adversarial training for signal interpretability.
