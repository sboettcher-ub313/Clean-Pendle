
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

class SyntheticRareEventGenerator:
    def __init__(self,
                 n_samples,
                 n_features,
                 imbalance_ratio=0.05,
                 noise_level=0.1,
                 concept_drift=0.0,
                 rare_event_weight=0.5,
                 seed=42):
        self.n_samples = n_samples
        self.n_features = n_features
        self.imbalance_ratio = imbalance_ratio
        self.noise_level = noise_level
        self.concept_drift = concept_drift
        self.rare_event_weight = rare_event_weight
        self.seed = seed

        # Derived internal parameters
        self.weights = [1 - imbalance_ratio, imbalance_ratio]
        self.n_informative = max(2, int(n_features * rare_event_weight))
        self.n_redundant = max(0, n_features - self.n_informative)
        self.class_sep = 1.5 - (rare_event_weight * 1.0)
        self.flip_y = noise_level

    def generate(self):
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=self.n_informative,
            n_redundant=self.n_redundant,
            weights=self.weights,
            class_sep=self.class_sep,
            flip_y=self.flip_y,
            random_state=self.seed,
        )
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(self.n_features)])
        df["rare_event"] = y
        return df

    def add_noise(self, df, noise_level=None):
        level = self.noise_level if noise_level is None else noise_level
        noise = np.random.normal(0, level, df.shape)
        df_noisy = df.copy()
        df_noisy.iloc[:, :-1] += noise[:, :-1]
        return df_noisy

    def inject_drift(self, df, drift_strength=None):
        strength = self.concept_drift if drift_strength is None else drift_strength
        drifted_df = df.copy()
        drifted_df.iloc[:, 0] += strength
        return drifted_df

    @staticmethod
    def save_to_csv(df, filepath):
        df.to_csv(filepath, index=False)

def generate_market_shock_dataset(difficulty="medium", random_state=42):
    config = {
        "easy":    {"imbalance_ratio": 0.1,  "noise_level": 0.1, "concept_drift": 0.1, "rare_event_weight": 0.3},
        "medium":  {"imbalance_ratio": 0.05, "noise_level": 0.2, "concept_drift": 0.3, "rare_event_weight": 0.5},
        "hard":    {"imbalance_ratio": 0.02, "noise_level": 0.3, "concept_drift": 0.6, "rare_event_weight": 0.8},
        "extreme": {"imbalance_ratio": 0.01, "noise_level": 0.5, "concept_drift": 1.0, "rare_event_weight": 0.9}
    }

    if difficulty not in config:
        raise ValueError(f"Unknown difficulty '{difficulty}'. Choose from {list(config.keys())}")

    params = config[difficulty]

    gen = SyntheticRareEventGenerator(
        n_samples=5000,
        n_features=15,
        imbalance_ratio=params["imbalance_ratio"],
        noise_level=params["noise_level"],
        concept_drift=params["concept_drift"],
        rare_event_weight=params["rare_event_weight"],
        seed=random_state
    )

    df = gen.generate()
    df = gen.add_noise(df)
    df = gen.inject_drift(df)
    return df
