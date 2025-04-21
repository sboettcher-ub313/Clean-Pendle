# 📦 param_mutator.py

import random

def mutate_params(params):
    """
    Slightly mutate the parameters for the next generation.
    Keeps values within valid bounds.
    """
    new_params = params.copy()

    # Mutate C (inverse of regularization strength)
    new_C = params["C"] * random.uniform(0.8, 1.2)
    new_params["C"] = max(0.01, min(new_C, 10.0))  # Keep it in a sensible range

    # Mutate l1_ratio (mix between L1 and L2 regularization)
    new_l1 = params["l1_ratio"] + random.uniform(-0.1, 0.1)
    new_params["l1_ratio"] = max(0.0, min(new_l1, 1.0))  # Must stay between 0 and 1

    return new_params