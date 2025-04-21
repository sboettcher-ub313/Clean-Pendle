# src/evaluation.py

import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve,
    auc, confusion_matrix, ConfusionMatrixDisplay
)

def evaluate_model(model, X_test, y_test, results=None, label=""):
    try:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        roc = roc_auc_score(y_test, y_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)

        if results is not None:
            results.append({
                "Method": label,
                "ROC AUC": roc,
                "PR AUC": pr_auc
            })

        print(f"\n--- {label} ---")
        print(classification_report(y_test, y_pred))
        print("ROC AUC:", round(roc, 3))
        print("PR AUC:", round(pr_auc, 3))

        plt.plot(recall, precision, label=f"{label} (PR AUC = {pr_auc:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid()
        plt.show()

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Event", "Rare Event"])
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f"Confusion Matrix: {label}")
        plt.grid(False)
        plt.show()

        return {"roc_auc": roc, "pr_auc": pr_auc}

    except Exception as e:
        print(f"[⚠️ Error evaluating {label}]:", e)
        return {"roc_auc": 0, "pr_auc": 0}