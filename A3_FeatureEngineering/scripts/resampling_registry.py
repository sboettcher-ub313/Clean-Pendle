from .resampling import (
    manual_upsampling, smote, adasyn, borderline_smote,
    smote_tomek, smote_enn, random_undersample, cluster_centroids
)

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