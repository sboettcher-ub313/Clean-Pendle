from sklearn.linear_model import LogisticRegression, SGDClassifier

def make_model(
    l1_ratio=0.5,
    C=1.0,
    max_iter=1000,
    tol=1e-2,
    class_weight='balanced',
    random_state=42,
    use_sgd=False,
    warm_start=False
):
    if use_sgd:
        return SGDClassifier(
            loss='log_loss',
            penalty='elasticnet',
            l1_ratio=l1_ratio,
            alpha=1.0 / C,  # inverse of C for SGD
            max_iter=max_iter,
            tol=tol,
            class_weight=class_weight,
            random_state=random_state,
            warm_start=warm_start
        )

    # fallback to LogisticRegression
    penalty = 'elasticnet'
    solver = 'saga'
    if l1_ratio in [0.0, 1.0]:
        solver = "liblinear"
        penalty = "l1" if l1_ratio == 1.0 else "l2"

    return LogisticRegression(
        penalty=penalty,
        solver=solver,
        l1_ratio=None if solver == "liblinear" else l1_ratio,
        C=C,
        max_iter=max_iter,
        tol=tol,
        class_weight=class_weight,
        random_state=random_state
    )

def make_child_model(params, max_iter=900, tol=1e-2, dataset_name=None, use_sgd=False, warm_start=False):
    l1_ratio = params.get("l1_ratio", 0.5)
    C = params.get("C", 1.0)

    # Boost settings for known tough datasets
    if dataset_name and "imbalanced_sparse" in dataset_name:
        max_iter = 1500
        tol = 5e-2  # Allow looser convergence

    return make_model(
        l1_ratio=l1_ratio,
        C=C,
        max_iter=max_iter,
        tol=tol,
        use_sgd=use_sgd,
        warm_start=warm_start
    )
