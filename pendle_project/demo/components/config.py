# demo/components/config.py

# ✅ Baseline config for synthetic data generation
BASE_CONFIG = {
    "imbalance": 0.1,
    "noise": 0.1,
    "drift": 0.1,
    "sparsity": 0.0,
    "confounding": 0.0,
    "shock": False,
    "cyclicality": 0.0,
    "cycle_freq": 0.1,
    "label_lag": 0,
    "n_features": 200,
    "redundant_ratio": 0.1,
    "useless_ratio": 0.1,
    "post_shock_dropout": 0.0,
    "reflexivity": 0.0,
    "burstiness": 0.0,
    "regime_complexity": 1,
    "semantic_drift": 0.0
}

# 🔁 Direct keyword → config impact
KEYWORD_MAP = {
    "layoffs": {"imbalance": 0.05, "drift": 0.1},
    "unemployment": {"imbalance": 0.06, "noise": 0.05},
    "war": {"shock": True, "burstiness": 0.4, "noise": 0.1},
    "geopolitical tension": {"drift": 0.15, "semantic_drift": 0.1},
    "yield curve": {"drift": 0.2, "noise": 0.1},
    "AI hype": {"noise": 0.25, "confounding": 0.3},
    "crypto collapse": {"shock": True, "burstiness": 0.3},
    "protests": {"burstiness": 0.2, "drift": 0.15},
    "fear index": {"noise": 0.2, "burstiness": 0.3, "semantic_drift": 0.1},
    "strike": {"burstiness": 0.2, "drift": 0.1},
    "inflation": {"noise": 0.15, "semantic_drift": 0.05},
    "tariffs": {"drift": 0.15, "semantic_drift": 0.1}
}

# 🧠 Aliases → canonical keywords
ALIASES = {
    "job cuts": "layoffs",
    "jobs": "unemployment",
    "bank run": "collapse",
    "trade war": "tariffs",
    "fed": "yield curve",
    "bitcoin": "crypto collapse",
    "demonstrations": "protests",
    "price surge": "inflation",
    "tarrifs": "tariffs",  # typo fallback
    "trump": "geopolitical tension",
}

# 💬 Fallback sentiment mappings
SENTIMENT_HINTS = {
    "fear": {"noise": 0.2, "semantic_drift": 0.2},
    "collapse": {"shock": True, "burstiness": 0.3},
    "optimism": {"imbalance": 0.3, "noise": 0.05},
    "tension": {"drift": 0.15},
    "fragile": {"noise": 0.1, "confounding": 0.15},
    "breakout": {"burstiness": 0.2, "semantic_drift": 0.1},
    "authoritarian": {"shock": True, "semantic_drift": 0.2},
    "populism": {"drift": 0.2, "reflexivity": 0.15}
}

##################
### CHECKPOINT ###
##################
# # config.py

# # 🔧 Generator default baseline
# BASE_SIM_CONFIG = {
#     "imbalance": 0.1,
#     "noise": 0.1,
#     "drift": 0.1,
#     "sparsity": 0.0,
#     "confounding": 0.0,
#     "shock": False,
#     "cyclicality": 0.0,
#     "cycle_freq": 0.1,
#     "label_lag": 0,
#     "n_features": 200,
#     "redundant_ratio": 0.1,
#     "useless_ratio": 0.1,
#     "post_shock_dropout": 0.0,
#     "reflexivity": 0.0,
#     "burstiness": 0.0,
#     "regime_complexity": 1,
#     "semantic_drift": 0.0,
#     "benford_violation": False,
#     "birthday_collisions": False
# }
 
# # 📁 Paths and folders
# PATHS = {
#     "assets": "demo/assets/",
#     "models": "models/",
#     "data": "data/",
#     "logs": "outputs/logs/"
# }

# # 🧠 Model choice
# DEFAULT_MODEL = {
#     "type": "logistic_regression",
#     "params": {
#         "solver": "liblinear",
#         "class_weight": "balanced"
#     }
# }

# # 🔍 Tuning & UI toggles
# UI_OPTIONS = {
#     "enable_pr_auc_display": True,
#     "max_keywords": 10
# }