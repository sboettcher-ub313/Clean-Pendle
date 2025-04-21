🎲 economic_predictor_playtime

This folder contains the core logic and creative modeling process behind our rare event predictor for synthetic economic data. We structure learning like play: training wheels, make-believe data, and gradual progression from naive guesses to interpretable causal diagnostics.

🧠 Purpose

To simulate realistic economic crises (e.g. market shocks, stress), run experiments on rare event prediction, and reverse-engineer what factors may have contributed—all without touching real data.

This approach matters because real-world economic data is messy, biased, and often limited by privacy or availability. By using synthetic datasets:
	•	We train safely — avoiding overfitting to history or revealing sensitive info.
	•	We simulate edge cases — rare events like crashes or systemic stress are underrepresented in historical data.
	•	We promote interpretability — reverse-engineering from synthetic causes helps build intuition, not just prediction.
	•	We democratize modeling — anyone (students, researchers, citizen scientists) can explore high-impact scenarios without barriers.

In short: synthetic training is play—with a purpose.

⸻

🧸 economic_predictor_playtime/

A whimsical yet rigorous space to:
	1.	Generate blinded synthetic economic datasets
	2.	Train stacker ensembles (ElasticNet, XGBoost)
	3.	Achieve PR AUC liftoff to detect market stress
	4.	Perform reverse PCA unblinding to understand “what features caused it?”

🧠 Notebooks:
	•	Step1_Initial_Exploration.ipynb: Blind runs with base models
	•	Step2_Liftoff_Baseline.ipynb: Proof-of-liftoff for rare events
	•	Step3_Lift_Booster.ipynb: Conservative boosting (avoiding overfit)
	•	Step4_Ensemble_Blender.ipynb: Logistic + XGBoost ensemble
	•	Step5_CausalDiagnostics.ipynb: PCA-based causal interpretation and mapping to economic concepts

📁 Datasets:
	•	market_shock_synthetic_datasets/: Simulated crisis scenarios
	•	“Features” are generic (feature_0, feature_1…), later reverse-mapped to concepts like GDP, CPI, PMI, ETF_flows, etc.

📘 Concept Pools:
	•	layman_friendly: “gas prices”, “consumer sentiment”
	•	professional_signal: “yield curve slope”, “credit spread”
	•	academic_theory: “output gap”, “natural rate of unemployment”
	•	model_outputs: “economic_surprise_index”, “nowcast_gdp”
	•	linchpins: “FED_FUNDS_RATE”, “GDP”, “UNEMPLOYMENT_RATE”

⸻

📂 Structure

economic_predictor_playtime/
├── champion_packages/               # Trained base models (ElasticNet, etc.)
├── champion_stacks/                 # Logistic regression stackers on base models
├── xgboost_stacks/                  # XGBoost stackers for boosting ensemble lift
├── market_shock_synthetic_datasets/ # Diverse generated datasets (easy → extreme)
├── archive/                         # (Optional) Legacy outputs and backups

# 🚀 Five Steps of Learning Progression
├── Step1_Initial_Exploration.ipynb     # Baseline models (blind guesswork)
├── Step2_Liftoff_Baseline.ipynb        # Breed-and-battle ElasticNet tournament
├── Step3_Lift_Booster.ipynb            # Boosting (XGBoost) after champion selection
├── Step4_Ensemble_Blender.ipynb        # Weighted blending of logistic + XGBoost
├── Step5_CausalDiagnostics.ipynb       # PCA unblinding to interpret rare event causes

# 📊 Results and Leaderboards
├── champion_cross_eval_results.csv     # All champion models on all datasets
├── ensemble_lift_leaderboard.csv       # Top stacker + blend model performance
├── ensemble_blended_lift_results.csv   # Final liftoff tracker
├── xgboost_lift_results.csv            # Boost-only benchmark

# 📘 Documentation
├── README.md                           # Project guide (you're here)
├── *.pdf                               # Notebook exports for presentation/review

⸻

📁 Notebook Guide

✅ Step 1: Step1_Initial_Exploration.ipynb

Goal: Evaluate naive models on synthetic rare event datasets.
	•	Introduces the concept of rare event detection
	•	Uses a variety of imbalanced, noisy, and overlapping synthetic datasets
	•	Measures model performance vs. a random baseline
👶 Like a child learning balance with training wheels

⸻

🚀 Step 2: Step2_Liftoff_Baseline.ipynb

Goal: Run a tournament where elastic net models evolve over generations to beat baseline PR AUC.
	•	Implements a “breed and battle” evolution system
	•	Shows how liftoff emerges: the first moments where models begin to consistently beat randomness
🧬 Models that “survive” are saved as champions for downstream use.

⸻

🔧 Step 3: Step3_Lift_Booster.ipynb

Goal: Carefully apply boosting methods like XGBoost to further improve performance
	•	Uses champion models from Step 2 as inputs
	•	Adds stacking (logistic + XGBoost) for smarter ensembling
🎯 Focuses on staying conservative—avoiding overfitting and checking for generalizability.

⸻

🧪 Step 4: Step4_Ensemble_Blender.ipynb

Goal: Create blended ensemble models that approach or exceed state-of-the-art performance
	•	Uses weighted blending of stackers
	•	Measures final lift over baseline across all datasets
🔥 This is where we hit our project target: PR AUC > 0.6 (SOTA liftoff threshold)

⸻

🔍 Step 5: Step5_CausalDiagnostics.ipynb

Goal: Reverse-engineer what scenarios caused rare events using PCA and economic concept mapping
	•	Blind and unblind: features are originally anonymized
	•	Dynamically assigns simulated economic labels
	•	Identifies which real-world signals (like “credit_spread” or “job_postings”) were most associated with rare outcomes
🔬 Helps humans learn from the machine’s point of view

⸻
