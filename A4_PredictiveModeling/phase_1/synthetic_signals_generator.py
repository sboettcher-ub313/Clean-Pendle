
# 📓 Synthetic Economic Signal Generator — MIMIC-Style Matrix

import numpy as np
import pandas as pd
import random
import os
from synthetic_generator import generate_signal_column
from MarketSimGenerators import generate_macro_feature, generate_market_feature

# --- Configuration ---
NUM_ROWS = 10000
NUM_FEATURES = 150
OUTPUT_FOLDER = "sim_data"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Concept Pools
CONCEPT_POOLS = {
    "macroeconomics": [
        "gdp_growth", "inflation_rate", "unemployment_rate",
        "consumer_sentiment", "interest_rate_policy"
    ],
    "markets": [
        "equity_volatility", "bond_yield_curve", "credit_spread",
        "market_liquidity", "foreign_exchange_rate"
    ],
    "commodities": [
        "oil_price", "commodity_index", "energy_supply_shock"
    ],
    "geopolitical": [
        "geopolitical_risk", "policy_uncertainty", "defense_spending"
    ],
    "engineered": [
        "latent_vector", "drift_signal", "noisy_blip", "spike_index"
    ]
}

# --- Generate Signals ---
all_features = []
signal_meta = {}

for i in range(NUM_FEATURES):
    pool_name = random.choice(list(CONCEPT_POOLS.keys()))
    concept_name = random.choice(CONCEPT_POOLS[pool_name])
    full_name = f"{pool_name}_{concept_name}_{i:03d}"

    # Choose generator
    if pool_name == "macroeconomics":
        col = generate_macro_feature(NUM_ROWS)
    elif pool_name == "markets":
        col = generate_market_feature(NUM_ROWS)
    else:
        col = generate_signal_column(NUM_ROWS)  # general fallback

    all_features.append(pd.Series(col, name=full_name))
    signal_meta[full_name] = {
        "pool": pool_name,
        "concept": concept_name,
        "transformation": "drift+noise+scaling"  # placeholder for now
    }

# --- Combine & Save ---
df = pd.concat(all_features, axis=1)
df.to_csv(os.path.join(OUTPUT_FOLDER, "synthetic_signals.csv"), index=False)

with open(os.path.join(OUTPUT_FOLDER, "signal_metadata.json"), "w") as f:
    json.dump(signal_meta, f, indent=2)

print("✅ Synthetic matrix + metadata exported.")
