from sklearn.linear_model import LogisticRegression

def make_model(
    l1_ratio=0.5,
    C=1.0,
    max_iter=900,
    tol=1e-2,
    class_weight='balanced',
    random_state=42
):
    return LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        l1_ratio=l1_ratio,
        C=C,
        max_iter=max_iter,
        tol=tol,
        class_weight=class_weight,
        random_state=random_state
    )

def make_child_model(params, max_iter=900, tol=1e-2, dataset_name=None):
    l1_ratio = params.get("l1_ratio", 0.5)
    C = params.get("C", 1.0)

    # ⏫ Boost settings for difficult datasets
    if dataset_name and "imbalanced_sparse" in dataset_name:
        max_iter = 1500
        tol = 5e-2

    return make_model(
        l1_ratio=l1_ratio,
        C=C,
        max_iter=max_iter,
        tol=tol
    )









# ###########################################################
# # LAST WORKING VERSION (only stage 5 struggled to converge)
# # 12:30PM Tue Apr 1
# from sklearn.linear_model import LogisticRegression

# def make_model(
#     penalty='elasticnet',
#     solver='saga',
#     l1_ratio=0.5,
#     C=1.0,
#     max_iter=200, # def make_model(..., max_iter=5000, ...) is overkill
#     tol=1e-2,
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

# # LAST WORKING: 9:07PM Mon Mar 31
# # def make_child_model(params):
# #     return make_model(
# #         l1_ratio=params.get("l1_ratio", 0.5),
# #         C=params.get("C", 1.0)
# #     )

# def make_child_model(params, max_iter=300):
#     return make_model(
#         l1_ratio=params.get("l1_ratio", 0.5),
#         C=params.get("C", 1.0),
#         max_iter=max_iter
#     )
# ###########################################################