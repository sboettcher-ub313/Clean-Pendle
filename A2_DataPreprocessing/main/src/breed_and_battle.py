import random
from sklearn.linear_model import LogisticRegression
import numpy as np
from model_eval import evaluate_model
# import matplotlib.pyplot as plt

# 🔧 Helper: mutate hyperparameters
def mutate_params(base_params, noise=0.1):
    return {
        "C": max(0.01, base_params["C"] + np.random.uniform(-noise, noise)),
        "l1_ratio": min(1.0, max(0.0, base_params["l1_ratio"] + np.random.uniform(-noise, noise)))
    }

# 🛠️ Helper: make logistic model
def make_child_model(params):
    return LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        l1_ratio=params["l1_ratio"],
        C=params["C"],
        max_iter=2000,
        tol=1e-4,
        class_weight='balanced',
        random_state=42
    )

# 🎙️ Commentary templates
child_comments = [
    "throws their gloves into the arena!", "marches in with fire!", "joins the fray!",
    "is ready to make waves!", "steps into the light!"
]
survivor_comments = [
    "clutches their spot for next round.", "barely makes the cut.",
    "dominates the leaderboard.", "hangs on by a thread.", "secures their legacy."
]
winner_comments = [
    "A new legend is born!", "The crowd goes wild!",
    "Undisputed champion!", "What a run!", "Glory secured!"
]
knockout_comments = [
    "is eliminated from the arena.", "falls short this time.", "can’t continue to the next round.",
    "is knocked out!", "is out of the tournament!"
]

# 🧬 Battle + Breeding Engine
def breed_and_battle(resampled_datasets, X_test, y_test, generations=3, top_k=3, debug=True):
    population = []

    if debug:
        print(f"\n🎬 Initial Fighters Enter the Arena!")

    for label, (X_res, y_res) in resampled_datasets.items():
        base_params = {"C": 1.0, "l1_ratio": 0.5}
        model = make_child_model(base_params)
        model.fit(X_res, y_res)
        score = evaluate_model(model, X_test, y_test, label=label, return_scores=True)
        population.append({
            "model": model,
            "params": base_params,
            "score": score,
            "label": label,
            "generation": 0,
            "lineage": f"origin of {label}"
        })

    for gen in range(1, generations + 1):
        if debug: print(f"\n🎮 Generation {gen} Begins!\n" + "-" * 25)

        sorted_pop = sorted(population, key=lambda x: x["score"]["pr_auc"], reverse=True)
        top_models = sorted_pop[:top_k]

        if debug:
            print("🏆 Top Performers:")
            for i, contender in enumerate(top_models, 1):
                print(f"{i}. {contender['label']}_G{contender['generation']}: PR AUC = {contender['score']['pr_auc']:.3f}")

        new_generation = []
        for parent in top_models:
            if debug: print(f"\n💘 {parent['label']} breeds 2 children")

            for child_num in range(1, 3):
                child_params = mutate_params(parent["params"])
                child_model = make_child_model(child_params)
                X_res, y_res = resampled_datasets[parent["label"]]
                child_model.fit(X_res, y_res)
                name = f"{parent['label']}_child_{child_num}_G{gen}"
                lineage = f"child of {parent['label']}_G{parent['generation']}"
                score = evaluate_model(child_model, X_test, y_test, label=name, return_scores=True)

                if debug:
                    print(f"\n👊 {name} {random.choice(child_comments)}")
                    print(f"📍 Style: {parent['label']}")
                    print(f"🧬 Genome: C = {child_params['C']:.2f}, l1_ratio = {child_params['l1_ratio']:.2f}")
                    print(f"🔁 Generation: {gen}")
                    print(f"🏆 Last Score: PR AUC = {score['pr_auc']:.3f}")
                    print(f"🧬 Lineage: {lineage}")

                new_generation.append({
                    "model": child_model,
                    "params": child_params,
                    "score": score,
                    "label": parent["label"],
                    "generation": gen,
                    "lineage": lineage
                })

        population.extend(new_generation)

        if debug:
            survivors = sorted(population, key=lambda x: x["score"]["pr_auc"], reverse=True)[:top_k * 3]
            survivor_ids = {id(s) for s in survivors}

            print(f"\n✅ Survivors advancing to next generation:")
            for s in survivors:
                comment = random.choice(survivor_comments)
                print(f"- {s['label']}_G{s['generation']} | PR AUC = {s['score']['pr_auc']:.3f} → {comment}")

            newly_created = [m for m in population if m["generation"] == gen]
            eliminated = [m for m in newly_created if id(m) not in survivor_ids]

            if eliminated:
                print(f"\n💀 Eliminated this round:")
                for elim in eliminated:
                    print(f"- {elim['label']}_G{elim['generation']} | PR AUC = {elim['score']['pr_auc']:.3f} → {random.choice(knockout_comments)}")

    final_winner = max(population, key=lambda x: x["score"]["pr_auc"])
    print("\n🏆 GRAND CHAMPION 🏆")
    print(f"👑 Name: {final_winner['label']}_G{final_winner['generation']}")
    print(f"🔁 Generation: {final_winner['generation']}")
    print(f"🧬 Genome: C = {final_winner['params']['C']:.2f}, l1_ratio = {final_winner['params']['l1_ratio']:.2f}")
    print(f"📈 PR AUC: {final_winner['score']['pr_auc']:.3f}")
    print(f"🎖 ROC AUC: {final_winner['score']['roc_auc']:.3f}")
    print(f"🧬 Lineage: {final_winner.get('lineage', 'Origin Unknown')}")
    print(f"🏁 {random.choice(winner_comments)}")

    return population