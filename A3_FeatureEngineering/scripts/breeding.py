import random
from models import make_model

def crossover(model_a, model_b):
    # Get parent hyperparameters
    l1_a, C_a = model_a["l1_ratio"], model_a["C"]
    l1_b, C_b = model_b["l1_ratio"], model_b["C"]
    
    # Simple average + noise
    l1_child = min(1.0, max(0.0, (l1_a + l1_b) / 2 + random.uniform(-0.1, 0.1)))
    C_child = max(0.01, (C_a + C_b) / 2 + random.uniform(-0.1, 0.1))

    return {
        "name": f"Child_L1{int(l1_child*100)}_C{str(round(C_child, 2)).replace('.', '')}",
        "model": make_model(l1_ratio=l1_child, C=C_child),
        "type": "elasticnet",
        "l1_ratio": l1_child,
        "C": C_child
    }

def mutate(model_cfg, mutation_rate=0.1):
    if random.random() > mutation_rate:
        return model_cfg

    new_l1 = min(1.0, max(0.0, model_cfg["l1_ratio"] + random.uniform(-0.2, 0.2)))
    new_C = max(0.01, model_cfg["C"] + random.uniform(-0.5, 0.5))

    return {
        "name": f"{model_cfg['name']}_mut",
        "model": make_model(l1_ratio=new_l1, C=new_C),
        "type": "elasticnet",
        "l1_ratio": new_l1,
        "C": new_C
    }