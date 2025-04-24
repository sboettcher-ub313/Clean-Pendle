import numpy as np
import pandas as pd
import time


def sigmoid_transition(length, slope=10):
    """
    Smooth transition curve used for regime change.
    Returns values between 0 and 1.
    """
    x = np.linspace(-1, 1, length)
    return 1 / (1 + np.exp(-slope * x))


def generate_synthetic_stress_dataset(
    config=None,
    n_samples=300,
    imbalance=0.1,
    noise=0.1,
    drift=0.1,
    sparsity=0.0,
    confounding=0.0,
    shock=False,
    cyclicality=0.0,
    cycle_freq=0.1,
    label_lag=0,
    n_features=200,
    redundant_ratio=0.1,
    useless_ratio=0.1,
    post_shock_dropout=0.0,
    reflexivity=0.0,
    burstiness=0.0,
    regime_complexity=1,
    semantic_drift=0.0,
    seed=None
):
    """
    Synthesizes a rare-event dataset mimicking stress signals across multiple latent regimes.
    """
    if config:
        locals().update(config)

    if seed is None:
        seed = int(time.time()) % 1_000_000
    np.random.seed(seed)

    # Feature counts
    true_features = int(n_features * (1 - redundant_ratio - useless_ratio))
    redundant_features = int(n_features * redundant_ratio)
    useless_features = n_features - true_features - redundant_features

    X = []
    transitions = sigmoid_transition(n_samples) if regime_complexity > 1 else np.ones(n_samples)

    for i in range(true_features):
        base = np.random.normal(0, 1, n_samples)

        if cyclicality > 0:
            base += cyclicality * np.sin(np.linspace(0, 2 * np.pi * cycle_freq, n_samples))

        if drift > 0:
            base += np.linspace(0, drift, n_samples)

        if reflexivity > 0:
            base += reflexivity * base

        if semantic_drift > 0:
            base += np.random.normal(0, semantic_drift, n_samples)

        # Apply regime complexity shaping
        base *= 1 + (regime_complexity - 1) * transitions

        X.append(base)

    # Redundant features
    for i in range(redundant_features):
        src = X[np.random.randint(0, len(X))]
        X.append(src + np.random.normal(0, 0.05, n_samples))

    # Useless features
    for i in range(useless_features):
        X.append(np.random.normal(0, 1, n_samples))

    X = np.vstack(X).T

    # Core rare event signal
    logits = X[:, 0] * 0.4 - X[:, 1] * 0.2 + (X[:, 2] * 0.1 if X.shape[1] > 2 else 0)
    logits += np.random.normal(0, noise, n_samples)
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > (1 - imbalance)).astype(int)

    # Burst / Shock injection
    if shock:
        burst_len = max(3, int(burstiness * 5))
        burst_start = np.random.randint(n_samples // 3, 2 * n_samples // 3)
        y[burst_start:burst_start + burst_len] = 1
        X[burst_start:burst_start + burst_len] += burstiness * 3

    # Confounding noise flips
    if confounding > 0:
        flip_idx = np.random.rand(n_samples) < confounding
        y[flip_idx] = 1 - y[flip_idx]

    # Sparsity: simulate dropout
    if sparsity > 0:
        X[np.random.rand(*X.shape) < sparsity] = np.nan

    # Label lag
    if label_lag > 0:
        y = np.roll(y, label_lag)
        y[:label_lag] = 0

    df = pd.DataFrame(X, columns=[f"signal_{i+1}" for i in range(X.shape[1])])
    df["rare_event"] = y

    return df
