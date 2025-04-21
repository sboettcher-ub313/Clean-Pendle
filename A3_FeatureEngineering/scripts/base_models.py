# LAST WORKING 9:39PM Mon Mar 31
# from models import make_model

# def get_base_models():
#     configs = [
#         {"l1_ratio": 0.3, "C": 0.1},
#         {"l1_ratio": 0.5, "C": 0.5},
#         {"l1_ratio": 0.7, "C": 1.0},
#         {"l1_ratio": 1.0, "C": 1.0},  # LASSO
#         {"l1_ratio": 0.0, "C": 1.0},  # Ridge
#     ]

#     models = []
#     for i, cfg in enumerate(configs):
#         model = make_model(
#             l1_ratio=cfg["l1_ratio"],
#             C=cfg["C"]
#         )
#         models.append({
#             "name": f"Elastic_L{int(cfg['l1_ratio']*100)}_C{int(cfg['C']*10)}",
#             "model": model
#         })
#     return models

from .core_models import make_model

def get_base_models():
    configs = [
        {"l1_ratio": 0.3, "C": 0.1},
        {"l1_ratio": 0.5, "C": 0.5},
        {"l1_ratio": 0.7, "C": 1.0},
        {"l1_ratio": 1.0, "C": 1.0},  # LASSO
        {"l1_ratio": 0.0, "C": 1.0},  # Ridge
    ]

    models = []
    for cfg in configs:
        model = make_model(
            l1_ratio=cfg["l1_ratio"],
            C=cfg["C"]
        )
        models.append({
            "name": f"Elastic_L{int(cfg['l1_ratio']*100)}_C{int(cfg['C']*10)}",
            "model": model,
            "params": cfg  # ✅ Added this line
        })
    return models