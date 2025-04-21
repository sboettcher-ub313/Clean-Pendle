# src/models.py

from sklearn.linear_model import LogisticRegression

def make_model(
    penalty='elasticnet',
    solver='saga',
    l1_ratio=0.5,
    C=1.0,
    max_iter=5000,
    tol=1e-4,
    class_weight='balanced',
    random_state=1
):
    """
    Returns a configured LogisticRegression model using ElasticNet regularization.
    
    Parameters can be adjusted to explore variations.
    """
    return LogisticRegression(
        penalty=penalty,
        solver=solver,
        l1_ratio=l1_ratio,
        C=C,
        max_iter=max_iter,
        tol=tol,
        class_weight=class_weight,
        random_state=random_state
    )