
from sklearn.linear_model import LogisticRegression

def train_predict_model(X, y):
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X.fillna(0), y)
    probs = model.predict_proba(X.fillna(0))[:, 1]
    return probs
