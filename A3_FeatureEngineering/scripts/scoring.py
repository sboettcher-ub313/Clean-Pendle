import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, roc_curve, confusion_matrix,
    auc
)

def score_model(y_true, y_prob):
    return {
        "pr_auc": average_precision_score(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }

def plot_pr_curve(y_true, y_prob, label="Model"):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.plot(recall, precision, label=label)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.legend()
    plt.show()

def plot_roc_curve(y_true, y_prob, label="Model"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, label=label)
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

def evaluate_model(model, X_test, y_test, label="model", return_scores=False):
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    roc = roc_auc_score(y_test, y_prob)

    if return_scores:
        return {"pr_auc": pr_auc, "roc_auc": roc}
    else:
        print(f"📈 {label} → PR AUC: {pr_auc:.3f}, ROC AUC: {roc:.3f}")