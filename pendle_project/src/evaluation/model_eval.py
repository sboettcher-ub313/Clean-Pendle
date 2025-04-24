
from sklearn.metrics import precision_recall_curve, auc

def compute_pr_auc(y_true, probs):
    precision, recall, _ = precision_recall_curve(y_true, probs)
    return auc(recall, precision)
