from sklearn.linear_model import LogisticRegression

#FORMER VERSION 10:02PM Mon Mar 31
# def make_model(
#     penalty='elasticnet',
#     solver='saga',
#     l1_ratio=0.5,
#     C=1.0,
#     max_iter=5000,
#     tol=1e-4,
#     class_weight='balanced',
#     random_state=42
# ):
#     return LogisticRegression(
#         penalty=penalty,
#         solver=solver,
#         l1_ratio=l1_ratio,
#         C=C,
#         max_iter=max_iter,
#         tol=tol,
#         class_weight=class_weight,
#         random_state=random_state
#     )
# ⚠️ Speed-optimized: relax convergence for faster tournament rounds
def make_child_model(params):
    return LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        l1_ratio=params["l1_ratio"],
        C=params["C"],
        max_iter=300,     # ⏱️ much faster
        tol=1e-3,         # 📉 looser convergence
        class_weight='balanced',
        random_state=42
    )

def make_child_model(params):
    return make_model(
        l1_ratio=params.get("l1_ratio", 0.5),
        C=params.get("C", 1.0)
    )
