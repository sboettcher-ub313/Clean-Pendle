import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

class SyntheticRareEventGenerator:
    def __init__(self, n_samples, n_features, n_informative, n_redundant,
                 class_sep=1.0, weights=[0.95, 0.05], flip_y=0.01, random_state=42):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_informative = n_informative
        self.n_redundant = n_redundant
        self.class_sep = class_sep
        self.weights = weights
        self.flip_y = flip_y
        self.random_state = random_state

    def generate(self):
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=self.n_informative,
            n_redundant=self.n_redundant,
            weights=self.weights,
            class_sep=self.class_sep,
            flip_y=self.flip_y,
            random_state=self.random_state,
        )
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(self.n_features)])
        df["rare_event"] = y
        return df

    def add_noise(self, df, noise_level=0.1):
        noise = np.random.normal(0, noise_level, df.shape)
        df_noisy = df.copy()
        df_noisy.iloc[:, :-1] += noise[:, :-1]  # only apply noise to features
        return df_noisy

    def inject_drift(self, df, drift_strength=0.3):
        drifted_df = df.copy()
        drifted_df.iloc[:, 0] += drift_strength  # Apply drift to one feature (e.g., feature_0)
        return drifted_df