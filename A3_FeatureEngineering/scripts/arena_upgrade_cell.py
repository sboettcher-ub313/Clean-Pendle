import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score

# === Arena Loader ===
def load_datasets(path="./synthetic_datasets/"):
    dataset_paths = sorted(glob.glob(os.path.join(path, "*.csv")))
    datasets = []
    for file in dataset_paths:
        df = pd.read_csv(file)
        X = df.drop("rare_event", axis=1)
        y = df["rare_event"]
        datasets.append((X, y, os.path.basename(file)))
    return datasets

# === Evaluation Across Arena ===
def evaluate_model_on_all_datasets(model, datasets):
    pr_aucs = []
    roc_aucs = []
    for X, y, name in datasets:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, stratify=y, test_size=0.3, random_state=42
            )
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]

            pr_auc = average_precision_score(y_test, y_prob)
            roc_auc = roc_auc_score(y_test, y_prob)

            pr_aucs.append(pr_auc)
            roc_aucs.append(roc_auc)

        except Exception as e:
            print(f"⚠️ Model failed on dataset {name}: {e}")
            continue

    return {
        "avg_pr_auc": sum(pr_aucs) / len(pr_aucs) if pr_aucs else 0.0,
        "avg_roc_auc": sum(roc_aucs) / len(roc_aucs) if roc_aucs else 0.0
    }