
# 🧠 A2_DataPreprocessing Project Hub

This repository supports structured experimentation in preparing financial and macroeconomic datasets for rare-event prediction and diagnostic modeling. It separates stable production pipelines from creative sandboxing for rapid prototyping and visual storytelling.

---

## 📁 Project Layout

```
A2_DataPreprocessing/
    A2_DataPreprocessing/
        .DS_Store
        sandbox_week3/
            .DS_Store
            config/
                neat_config2.txt
                neat_config3.txt
            images/
                .DS_Store
                references/
                outputs/
                    Diffusion Model Confidence in Predictions.png
                    SGD Screenshot 2025-03-26 at 3.47.43ΓÇ»PM.png
                    SGD Screenshot 2025-03-26 at 3.48.02ΓÇ»PM.png
                    animated_diffusion_pca.gif
                    cnn_confidence_histogram.png
                    cnn_confusion_matrix.png
                    cnn_diagnostics_dashboard.png
                    cnn_diagnostics_dashboard_labeled.png
                    cnn_diagnostics_dashboard_labeled_large.png
                    cnn_diagnostics_dashboard_polished.png
                    cnn_loss_curve.png
                    cnn_metrics_table.png
                    cnn_roc_curve.png
                    diffusion_clusters_pca.png
                    diffusion_diagnostics_dashboard.png
                    diffusion_diagnostics_dashboard_labeled.png
                    elastic_net_results.png
                    gradient_boosting_real_performance.png
                    roc_curve_elastic_net.gif
                    roc_curve_elastic_net.png
                    roc_curve_gradient_boosting.gif
                    sgd_combined_performance.png
                    sgd_performance.png
                    sgd_real_performance.png
                    sgd_synthetic_performance.png
                    zscores.png
                    ≡ƒö╣ Market Stress Over Time & Top Features.png
            models/
                .DS_Store
                saved/
                    CNN (MLP).pkl
                    Diffusion_Model_GMM.pkl
                    Elastic Net.pkl
                    GA-Optimized LR.pkl
                    GradientBoosting.pkl
                    NeuroEvolution (NEAT).pkl
                    SGD.pkl
                    StandardScaler.pkl
                    gradient_boosting_scaler.pkl
                    selected_features.pkl
                    sgd_scaler.pkl
                metrics/
                    .DS_Store
                    H-(((S-12 x C-7) x (C-6 x G-4)) x N-3) (Gen 10)_metadata.json
                    Model_Performance_Metrics.csv
                    Model_Performance_Metrics_20250326_025015.csv
                    Model_Performance_Metrics_20250326_025300.csv
                    Model_Performance_Metrics_20250326_025529.csv
                    Model_Performance_Metrics_20250326_025847.csv
                    Model_Performance_Metrics_20250326_201722.csv
                    Model_Performance_Metrics_20250326_201912.csv
                    Model_Performance_Metrics_20250326_202020.csv
                    Model_Performance_Metrics_20250326_202036.csv
                    Model_Performance_Metrics_20250326_202935.csv
                    Model_Performance_Metrics_20250326_203320.csv
                    Model_Performance_Metrics_20250326_203401.csv
                    Model_Performance_Metrics_20250326_203436.csv
                    Model_Performance_Metrics_20250326_203930.csv
                    Model_Performance_Metrics_20250326_205027.csv
                    Model_Performance_Metrics_20250326_205039.csv
                    Model_Performance_Metrics_20250326_205124.csv
                    Model_Performance_Metrics_20250326_205142.csv
                    Model_Performance_Metrics_20250326_205659.csv
                    Model_Performance_Metrics_20250326_210240.csv
                    Model_Performance_Metrics_20250326_211151.csv
                    Model_Performance_Metrics_20250326_211743.csv
                    Model_Performance_Metrics_20250326_214329.csv
                    Model_Performance_Metrics_20250326_214606.csv
                    Model_Performance_Metrics_20250326_215324.csv
                    Model_Performance_Metrics_20250326_220534.csv
                    Model_Performance_Metrics_20250326_220549.csv
                    Model_Performance_Metrics_20250326_220910.csv
                    Model_Performance_Metrics_20250326_222224.csv
                    Model_Performance_Metrics_Gen2.csv
                    Model_Performance_Metrics_Updated.csv
                    NeuroEvolution (NEAT)_metadata.json
                    S-12_metadata.json
                    feature_names.json
                    model_performance.csv
                trained/
                    CNN (MLP).pkl
                    Diffusion Model.pkl
                    Elastic Net.pkl
                    GA-Optimized LR.pkl
                    Gen10_Model0.pkl
                    Gen10_Model0_20250326_022722.pkl
                    Gen10_Model0_20250326_023438.pkl
                    Gen1_Model0_20250326_022722.pkl
                    Gen1_Model0_20250326_023438.pkl
                    Gen1_Model1_20250326_023438.pkl
                    Gen1_Model2_20250326_023438.pkl
                    Gen1_Model3_20250326_023438.pkl
                    Gen2_Model0_20250326_022722.pkl
                    Gen2_Model0_20250326_023438.pkl
                    Gen2_Model1_20250326_023438.pkl
                    Gen3_Model0_20250326_022722.pkl
                    Gen3_Model0_20250326_023438.pkl
                    Gen4_Model0_20250326_022722.pkl
                    Gen4_Model0_20250326_023438.pkl
                    Gen5_Model0_20250326_022722.pkl
                    Gen5_Model0_20250326_023438.pkl
                    Gen6_Model0_20250326_022722.pkl
                    Gen6_Model0_20250326_023438.pkl
                    Gen7_Model0_20250326_022722.pkl
                    Gen7_Model0_20250326_023438.pkl
                    Gen8_Model0_20250326_022722.pkl
                    Gen8_Model0_20250326_023438.pkl
                    Gen9_Model0_20250326_022722.pkl
                    Gen9_Model0_20250326_023438.pkl
                    Gradient Boosting.pkl
                    NeuroEvolution (NEAT).pkl
                    SGD.pkl
                    Supermodel.pkl
                    model_performance.pkl
                logs/
                    H-(((S-12 x C-7) x (C-6 x G-4)) x N-3) (Gen 10).pt
                .ipynb_checkpoints/
                    Model_Performance_Metrics-checkpoint.csv
                    Model_Performance_Metrics_20250326_025015-checkpoint.csv
                    Model_Performance_Metrics_20250326_025300-checkpoint.csv
                    Model_Performance_Metrics_20250326_025529-checkpoint.csv
                    Model_Performance_Metrics_20250326_025847-checkpoint.csv
                    Model_Performance_Metrics_Updated-checkpoint.csv
                    model_performance-checkpoint.csv
            exports/
            scripts/
            data/
                .DS_Store
                logs/
                    .DS_Store
                    cnn_results.csv
                    diffusion_gmm_results.csv
                    elastic_net_results.csv
                    evolution_log.txt
                    gradient_boosting_results.csv
                    model_performance copy.csv
                    model_performance_universal.csv
                    perfmodel_performance_universal.csv
                    sgd_results.csv
                synthetic/
                    .DS_Store
                    Synthetic_Rare_Event_Dataset.csv
                    synth_findata.csv
                    synth_rare_event_data.csv
                external/
                processed/
                    financial_data_btc_era.csv
                    financial_data_cleaned.csv
                    financial_data_cleaned2.csv
                    financial_data_full.csv
                    financial_data_pca.csv
                    financial_data_scaled.csv
                    market_stress.csv
                raw/
            notebooks/
                .DS_Store
                BreedAndBattleRoyale.ipynb
                Spellbook2.ipynb
                Untitled.ipynb
                Untitled2.ipynb
                battleandbreed.ipynb
                cnn_mlp.ipynb
                diffusion.ipynb
                donerkebabandblitzen.ipynb
                elastic.ipynb
                galr.ipynb
                gradientboost.ipynb
                jank.ipynb
                makingbacon.ipynb
                neat.ipynb
                pinkpowerpuff copy.ipynb
                pinkpowerpuff.ipynb
                pipeline_A_manual_upsampling_elasticnet.ipynb
                pipeline_B_smote_elasticnet.ipynb
                pipeline_C_smotetomek_elasticnet.ipynb
                plslive.ipynb
                rare_event_resampling_experiments-Copy1.ipynb
                rare_event_resampling_experiments.ipynb
                rare_event_resampling_experiments2.ipynb
                sgd.ipynb
                smote.ipynb
                synthehol.ipynb
                temp.ipynb
                test.ipynb
                universal_rare_event_pipeline.ipynb
                viibe.ipynb
        main/
            .DS_Store
            README.md
            requirements.txt
            pics/
                image.png
                .ipynb_checkpoints/
                    image-checkpoint.png
                sprites/
                    blob_adasyn.png
                    blob_borderline_smote.png
                    blob_cluster_centroids.png
                    blob_manual_upsampling.png
                    blob_no_resampling.png
                    blob_random_undersample.png
                    blob_smote.png
                    blob_smoteenn.png
                    blob_smotetomek.png
                    .ipynb_checkpoints/
                        blob_borderline_smote-checkpoint.png
                        blob_cluster_centroids-checkpoint.png
                        blob_manual_upsampling-checkpoint.png
            .ipynb_checkpoints/
                README-checkpoint.md
                requirements-checkpoint.txt
            data/
                .DS_Store
                synth_rare_event_data.csv
            notebooks/
                .DS_Store
                RareEvent_BreedAndBattle_WithInsights.ipynb
                battle_arena.html
                battle_log.txt
                battle_log_data.js
                battle_template.html
                rare_event_analysis.ipynb
                archive/
                    BattleRoyale.ipynb
                    BattleRoyale_Evolution_20250330_150414.ipynb
                    RareEvent_BreedAndBattle_Optimized.ipynb
                    RareEvent_Evolution_Updated.ipynb
                    Untitled.ipynb
                    battle_arena.html
                    breed_and_battle_royale.ipynb
                    rare_event_breed_and_battle.ipynb
                    .ipynb_checkpoints/
                        BattleRoyale-checkpoint.ipynb
                        BattleRoyale_Evolution_20250330_150414-checkpoint.ipynb
                        RareEvent_BreedAndBattle_Optimized-checkpoint.ipynb
                        RareEvent_Evolution_Updated-checkpoint.ipynb
                        Untitled-checkpoint.ipynb
                        breed_and_battle_royale-checkpoint.ipynb
                        rare_event_breed_and_battle-checkpoint.ipynb
                .ipynb_checkpoints/
                    RareEvent_BreedAndBattle_WithInsights-checkpoint.ipynb
                    analysis2-checkpoint.ipynb
                    battle_arena-Copy1-checkpoint.html
                    battle_log-checkpoint.txt
                    rare_event_analysis-checkpoint.ipynb
            src/
                .DS_Store
                __init__.py
                battle_logger.py
                breed_and_battle.py
                evaluation.py
                model_eval.py
                models.py
                preprocessing.py
                resampling.py
                resampling_registry.py
                visualization.py
                __pycache__/
                    battle_logger.cpython-311.pyc
                    breed_and_battle.cpython-311.pyc
                    evaluation.cpython-311.pyc
                    model_eval.cpython-311.pyc
                    models.cpython-311.pyc
                    preprocessing.cpython-311.pyc
                    resampling.cpython-311.pyc
                    resampling_registry.cpython-311.pyc
                    visualization.cpython-311.pyc
                .ipynb_checkpoints/
                    __init__-checkpoint.py
                    battle_logger-checkpoint.py
                    breed_and_battle-checkpoint.py
                    evaluation-checkpoint.py
                    model_eval-checkpoint.py
                    models-checkpoint.py
                    preprocessing-checkpoint.py
                    resampling-checkpoint.py
                    resampling_registry-checkpoint.py
                    visualization-checkpoint.py
    __MACOSX/
        A2_DataPreprocessing/
            ._.DS_Store
            sandbox_week3/
                ._.DS_Store
                config/
                    ._neat_config2.txt
                    ._neat_config3.txt
                images/
                    ._.DS_Store
                    outputs/
                        ._SGD Screenshot 2025-03-26 at 3.47.43ΓÇ»PM.png
                        ._SGD Screenshot 2025-03-26 at 3.48.02ΓÇ»PM.png
                        ._≡ƒö╣ Market Stress Over Time & Top Features.png
                models/
                    ._.DS_Store
                    metrics/
                        ._.DS_Store
                        ._Model_Performance_Metrics_20250326_025300.csv
                        ._NeuroEvolution (NEAT)_metadata.json
                        ._S-12_metadata.json
                    logs/
                        ._H-(((S-12 x C-7) x (C-6 x G-4)) x N-3) (Gen 10).pt
                data/
                    ._.DS_Store
                    logs/
                        ._.DS_Store
                        ._evolution_log.txt
                    synthetic/
                        ._.DS_Store
                    processed/
                        ._market_stress.csv
                notebooks/
                    ._.DS_Store
                    ._smote.ipynb
            main/
                ._.DS_Store
                ._README.md
                ._requirements.txt
                data/
                    ._.DS_Store
                notebooks/
                    ._.DS_Store
                    ._battle_arena.html
                    ._battle_log.txt
                    ._battle_log_data.js
                    ._battle_template.html
                    archive/
                        ._battle_arena.html
                src/
                    ._.DS_Store
```

---

## 🗂️ Folder Roles

### `/main/`
- **Purpose**: Validated, reproducible pipeline for data processing and modeling.
- **Includes**: Core notebooks, cleaned data, stable outputs.

### `/sandbox_week3/`
- **Purpose**: Flexible experimentation zone for week 3 developments.
- **Includes**:
  - NEAT configs and genetic architecture
  - CNN, Diffusion, and ElasticNet model experiments
  - Diagnostic dashboards and advanced plots
  - Trained model checkpoints
  - CSV logs of model metrics and performance breakdowns
  - Animated PCA clusters and ROC curves

---

## 🧭 How to Use This Repository

### Development Flow

1. Prototype in `sandbox_week3/`
2. Document results and save visual outputs
3. Promote working pipelines or assets into `main/` for stability

### Suggested Practice

- Keep `main/` clean: Only add reproducible assets
- Use `sandbox_week3/` like a lab notebook
- Store experimental model configs under `sandbox_week3/config/`
- Export all model results to `metrics/` and visualizations to `images/outputs/`

---

## 🔍 Feature Highlights

- Supports evolutionary NEAT, CNN (MLP), Diffusion, ElasticNet, SGD, and Gradient Boosting
- Saves performance metrics for multiple timestamps and configurations
- Robust diagnostics including:
  - Confusion matrices
  - Loss curves
  - ROC curves
  - Animated PCA/GIF clusters
- Clean separation between `saved/` (raw models) and `trained/` (final picks)

---

## 🧾 Documentation Pointers

Each subfolder should ideally contain:
- `README.md` for local structure
- `.gitignore` if models or logs are heavy or constantly changing
- `requirements.txt` for env recreation (if needed)

---

## 👥 Collaboration Guide

- Use `sandbox_week3/README.md` for weekly changelogs, insights, and narrative
- Consider archiving older weeks to `sandbox_week2/`, `sandbox_week1/` as needed
- Prefer readable filenames for visual output (what, which model, when)

---

## 👋 Maintainer

Curated and maintained by **Sophia Boettcher**, 2025  
Inspired by clinical diagnostics, economic foresight, and ML transparency.

