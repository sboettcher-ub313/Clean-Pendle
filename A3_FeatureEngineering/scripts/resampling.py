import pandas as pd
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids

# -------------------------
# 💡 Individual Resamplers
# -------------------------

def manual_upsampling(X_train, y_train, target_col="rare_event"):
    df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    minority = df[df[target_col] == 1]
    majority = df[df[target_col] == 0]

    minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    upsampled = pd.concat([majority, minority_upsampled]).sample(frac=1, random_state=42).reset_index(drop=True)

    return upsampled.drop(columns=[target_col]), upsampled[target_col]

def smote(X, y):
    return SMOTE(random_state=42).fit_resample(X, y)

def adasyn(X, y):
    return ADASYN(random_state=42).fit_resample(X, y)

def borderline_smote(X, y):
    return BorderlineSMOTE(random_state=42).fit_resample(X, y)

def smote_tomek(X, y):
    return SMOTETomek(random_state=42).fit_resample(X, y)

def smote_enn(X, y):
    return SMOTEENN(random_state=42).fit_resample(X, y)

def random_undersample(X, y):
    return RandomUnderSampler(random_state=42).fit_resample(X, y)

def cluster_centroids(X, y):
    return ClusterCentroids(random_state=42).fit_resample(X, y)

# -------------------------------
# 🔁 Wrapper for Batch Evaluation
# -------------------------------

def get_resamplers(X_train, y_train, target_col="rare_event"):
    return {
        "Manual Upsampling": lambda: manual_upsampling(X_train, y_train, target_col),
        "SMOTE": lambda: smote(X_train, y_train),
        "ADASYN": lambda: adasyn(X_train, y_train),
        "Borderline SMOTE": lambda: borderline_smote(X_train, y_train),
        "SMOTETomek": lambda: smote_tomek(X_train, y_train),
        "SMOTEENN": lambda: smote_enn(X_train, y_train),
        "Random Undersample": lambda: random_undersample(X_train, y_train),
        "Cluster Centroids": lambda: cluster_centroids(X_train, y_train),
        "No Resampling": lambda: (X_train, y_train)
    }