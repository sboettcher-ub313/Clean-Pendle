# Shortened version for brevity — same as provided in previous message
# Contains: mutate_params, make_child_model, breed_and_battle_with_population
# Uses: model_cfg["params"] and build model inside breed_and_battle_with_population
# Fixes: Missing 'model' key issue, aligns with how survivors are rebuilt between stages
import random
import numpy as np
import gc
from sklearn.linear_model import LogisticRegression
from models.model_eval import evaluate_model
from models import make_child_model

# 🔧 Mutate hyperparameters slightly to simulate genetic variation
def mutate_params(base_params, noise=0.1):
    return {
        "C": max(0.01, base_params["C"] + np.random.uniform(-noise, noise)),
        "l1_ratio": min(1.0, max(0.0, base_params["l1_ratio"] + np.random.uniform(-noise, noise)))
    }

# 🎙️ Style commentary
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

# import gc
import pandas as pd
import random
# from models.core_models import make_child_model
from utils.scoring import evaluate_model
from utils.battle_comments import child_comments, survivor_comments, knockout_comments, winner_comments
from utils.param_mutator import mutate_params  # Assuming you have this for mutation logic

def breed_and_battle_with_population(
    model_population,
    resampled_datasets,
    X_test,
    y_test,
    generations=3,
    top_k=3,
    debug=True,
    dataset_name="unknown",
    baseline_pr_auc=0.0
):
    population = []

    if debug:
        print(f"\n🎬 Initial Fighters Enter the Arena!")

    # 🧪 Round 0: Evaluate all starting models
    for model_cfg in model_population:
        model_name = model_cfg["name"]
        base_params = model_cfg.get("params", {"C": 1.0, "l1_ratio": 0.5})

        for resampler_label, (X_res, y_res) in resampled_datasets.items():
            label = f"{model_name} + {resampler_label}"
            model = make_child_model(base_params)
            model.fit(X_res, y_res)
            score = evaluate_model(model, X_test, y_test, label=label, return_scores=True)

            population.append({
                "params": base_params,
                "score": score,
                "label": label,
                "generation": 0,
                "lineage": f"origin of {label}",
                "dataset_name": dataset_name,
                "baseline_pr_auc": baseline_pr_auc
            })

            del model, X_res, y_res
            gc.collect()

    # 🔁 Evolution loop
    for gen in range(1, generations + 1):
        if debug:
            print(f"\n🎮 Generation {gen} Begins!\n" + "-" * 25)

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

                try:
                    resampler_label = parent["label"].split(" + ")[1]
                    X_res, y_res = resampled_datasets[resampler_label]
                except (IndexError, KeyError):
                    print(f"[⚠️] Couldn't locate resampler for {parent['label']}, skipping child.")
                    continue

                child_model.fit(X_res, y_res)
                name = f"{parent['label']}_child_{child_num}_G{gen}"
                lineage = f"child of {parent['label']}_G{parent['generation']}"
                score = evaluate_model(child_model, X_test, y_test, label=name, return_scores=True)

                new_generation.append({
                    "params": child_params,
                    "score": score,
                    "label": parent["label"],
                    "generation": gen,
                    "lineage": lineage,
                    "dataset_name": dataset_name,
                    "baseline_pr_auc": baseline_pr_auc
                })

                if debug:
                    print(f"\n👊 {name} {random.choice(child_comments)}")
                    print(f"📍 Style: {resampler_label}")
                    print(f"🧬 Genome: C = {child_params['C']:.2f}, l1_ratio = {child_params['l1_ratio']:.2f}")
                    print(f"🔁 Generation: {gen}")
                    print(f"🏆 PR AUC = {score['pr_auc']:.3f}")
                    print(f"🧬 Lineage: {lineage}")

                del child_model, X_res, y_res
                gc.collect()

        # 📉 Prune and retain only best models (parents + children)
        population = sorted(population + new_generation, key=lambda x: x["score"]["pr_auc"], reverse=True)[:top_k * 3]

        if debug:
            survivors = population[:top_k * 3]
            survivor_ids = {id(s) for s in survivors}
            newly_created = [m for m in new_generation]

            print(f"\n✅ Survivors advancing to next generation:")
            for s in survivors:
                comment = random.choice(survivor_comments)
                print(f"- {s['label']}_G{s['generation']} | PR AUC = {s['score']['pr_auc']:.3f} → {comment}")

            if newly_created:
                eliminated = [m for m in newly_created if id(m) not in survivor_ids]
                if eliminated:
                    print(f"\n💀 Eliminated this round:")
                    for elim in eliminated:
                        print(f"- {elim['label']}_G{elim['generation']} | PR AUC = {elim['score']['pr_auc']:.3f} → {random.choice(knockout_comments)}")

            # Export CSV summary (optional — comment out to reduce I/O)
            summary_table = [{
                "Generation": model["generation"],
                "Dataset": dataset_name,
                "Baseline PR AUC": round(baseline_pr_auc, 3),
                "Label": f"{model['label']}_G{model['generation']}",
                "PR AUC": round(model["score"]["pr_auc"], 3),
                "Beats Baseline?": "✅ Yes" if model["score"]["pr_auc"] > baseline_pr_auc else "❌ No",
                "Lineage": model.get("lineage", "—"),
                "Status": "✅ Survived" if id(model) in survivor_ids else "❌ Eliminated"
            } for model in newly_created]

            df_summary = pd.DataFrame(summary_table).sort_values(by=["Generation", "PR AUC"], ascending=[True, False])
            print(f"\n📋 Summary Table – Generation {gen} (Dataset: {dataset_name}):\n")
            print(df_summary)

            # Optional: only write summary file on final generation
            if gen == generations:
                df_summary.to_csv(f"logs/generation_{gen}_{dataset_name}_summary.csv", index=False)

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












# LAST WORKING: 12:14PM Tue Apr 1
# 🧬 Evolutionary battle simulation with population
# def breed_and_battle_with_population(model_population, resampled_datasets, X_test, y_test, generations=3, top_k=3, debug=True, dataset_name="unknown", baseline_pr_auc=0.0):
#     import pandas as pd
#     population = []

#     if debug:
#         print(f"\n🎬 Initial Fighters Enter the Arena!")

#     for model_cfg in model_population:
#         model_name = model_cfg["name"]
#         base_params = model_cfg.get("params", {"C": 1.0, "l1_ratio": 0.5})
#         for resampler_label, (X_res, y_res) in resampled_datasets.items():
#             label = f"{model_name} + {resampler_label}"
#             model = make_child_model(base_params)
#             model.fit(X_res, y_res)
#             score = evaluate_model(model, X_test, y_test, label=label, return_scores=True)

#             population.append({
#                 "model": model,
#                 "params": base_params,
#                 "score": score,
#                 "label": label,
#                 "generation": 0,
#                 "lineage": f"origin of {label}",
#                 "dataset_name": dataset_name,
#                 "baseline_pr_auc": baseline_pr_auc
#             })

#     for gen in range(1, generations + 1):
#         if debug: print(f"\n🎮 Generation {gen} Begins!\n" + "-" * 25)

#         sorted_pop = sorted(population, key=lambda x: x["score"]["pr_auc"], reverse=True)
#         top_models = sorted_pop[:top_k]

#         if debug:
#             print("🏆 Top Performers:")
#             for i, contender in enumerate(top_models, 1):
#                 print(f"{i}. {contender['label']}_G{contender['generation']}: PR AUC = {contender['score']['pr_auc']:.3f}")

#         new_generation = []
#         for parent in top_models:
#             if debug: print(f"\n💘 {parent['label']} breeds 2 children")

#             for child_num in range(1, 3):
#                 child_params = mutate_params(parent["params"])
#                 child_model = make_child_model(child_params)

#                 try:
#                     resampler_label = parent["label"].split(" + ")[1]
#                     X_res, y_res = resampled_datasets[resampler_label]
#                 except (IndexError, KeyError):
#                     print(f"[⚠️] Couldn't locate resampler for {parent['label']}, skipping child.")
#                     continue

#                 child_model.fit(X_res, y_res)
#                 # training speedup
#                 del X_res, y_res, child_model, child_params
#                 gc.collect()
                
#                 name = f"{parent['label']}_child_{child_num}_G{gen}"
#                 lineage = f"child of {parent['label']}_G{parent['generation']}"
#                 score = evaluate_model(child_model, X_test, y_test, label=name, return_scores=True)

#                 if debug:
#                     print(f"\n👊 {name} {random.choice(child_comments)}")
#                     print(f"📍 Style: {resampler_label}")
#                     print(f"🧬 Genome: C = {child_params['C']:.2f}, l1_ratio = {child_params['l1_ratio']:.2f}")
#                     print(f"🔁 Generation: {gen}")
#                     print(f"🏆 Last Score: PR AUC = {score['pr_auc']:.3f}")
#                     print(f"🧬 Lineage: {lineage}")

#                 new_generation.append({
#                     "model": child_model,
#                     "params": child_params,
#                     "score": score,
#                     "label": parent["label"],
#                     "generation": gen,
#                     "lineage": lineage,
#                     "dataset_name": dataset_name,
#                     "baseline_pr_auc": baseline_pr_auc
#                 })

#         # population.extend(new_generation)
#         # we are only keeping survivors of each generation
#         # limit memory to just the best models, avoiding runaway object growth
#         population = sorted(population + new_generation, key=lambda x: x["score"]["pr_auc"], reverse=True)[:top_k * 3]

#         if debug:
#             survivors = sorted(population, key=lambda x: x["score"]["pr_auc"], reverse=True)[:top_k * 3]
#             survivor_ids = {id(s) for s in survivors}

#             print(f"\n✅ Survivors advancing to next generation:")
#             for s in survivors:
#                 comment = random.choice(survivor_comments)
#                 print(f"- {s['label']}_G{s['generation']} | PR AUC = {s['score']['pr_auc']:.3f} → {comment}")

#             newly_created = [m for m in population if m["generation"] == gen]
#             eliminated = [m for m in newly_created if id(m) not in survivor_ids]

#             if eliminated:
#                 print(f"\n💀 Eliminated this round:")
#                 for elim in eliminated:
#                     print(f"- {elim['label']}_G{elim['generation']} | PR AUC = {elim['score']['pr_auc']:.3f} → {random.choice(knockout_comments)}")

#             summary_table = []
#             for model in newly_created:
#                 pr_auc = model["score"]["pr_auc"]
#                 summary_table.append({
#                     "Generation": model["generation"],
#                     "Dataset": dataset_name,
#                     "Baseline PR AUC": round(baseline_pr_auc, 3),
#                     "Label": f"{model['label']}_G{model['generation']}",
#                     "PR AUC": round(pr_auc, 3),
#                     "Beats Baseline?": "✅ Yes" if pr_auc > baseline_pr_auc else "❌ No",
#                     "Lineage": model.get("lineage", "—"),
#                     "Status": "✅ Survived" if id(model) in survivor_ids else "❌ Eliminated"
#                 })

#             df_summary = pd.DataFrame(summary_table).sort_values(by=["Generation", "PR AUC"], ascending=[True, False])
#             print(f"\n📋 Summary Table – Generation {gen} (Dataset: {dataset_name}):\n")
#             print(df_summary)
#             df_summary.to_csv(f"logs/generation_{gen}_{dataset_name}_summary.csv", index=False)

#     final_winner = max(population, key=lambda x: x["score"]["pr_auc"])
#     print("\n🏆 GRAND CHAMPION 🏆")
#     print(f"👑 Name: {final_winner['label']}_G{final_winner['generation']}")
#     print(f"🔁 Generation: {final_winner['generation']}")
#     print(f"🧬 Genome: C = {final_winner['params']['C']:.2f}, l1_ratio = {final_winner['params']['l1_ratio']:.2f}")
#     print(f"📈 PR AUC: {final_winner['score']['pr_auc']:.3f}")
#     print(f"🎖 ROC AUC: {final_winner['score']['roc_auc']:.3f}")
#     print(f"🧬 Lineage: {final_winner.get('lineage', 'Origin Unknown')}")
#     print(f"🏁 {random.choice(winner_comments)}")

#     return population







### copied over so the earlier versions still run
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

# forked from breed_and_battle
def breed_and_battle_with_population(model_population, resampled_datasets, X_test, y_test, generations=3, top_k=3, debug=True, dataset_name="unknown", baseline_pr_auc=0.0):
    import pandas as pd
    population = []

    if debug:
        print(f"\n🎬 Initial Fighters Enter the Arena!")

    for model_cfg in model_population:
        model_name = model_cfg["name"]
        for resampler_label, (X_res, y_res) in resampled_datasets.items():
            label = f"{model_name} + {resampler_label}"
            model = model_cfg["model"]
            model.fit(X_res, y_res)
            score = evaluate_model(model, X_test, y_test, label=label, return_scores=True)

            population.append({
                "model": model,
                "params": model_cfg.get("params", {"C": 1.0, "l1_ratio": 0.5}),
                "score": score,
                "label": label,
                "generation": 0,
                "lineage": f"origin of {label}",
                "dataset_name": dataset_name,            # ✅ NEW
                "baseline_pr_auc": baseline_pr_auc       # ✅ NEW
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

                try:
                    resampler_label = parent["label"].split(" + ")[1]
                    X_res, y_res = resampled_datasets[resampler_label]
                except (IndexError, KeyError):
                    print(f"[⚠️] Couldn't locate resampler for {parent['label']}, skipping child.")
                    continue

                child_model.fit(X_res, y_res)
                name = f"{parent['label']}_child_{child_num}_G{gen}"
                lineage = f"child of {parent['label']}_G{parent['generation']}"
                score = evaluate_model(child_model, X_test, y_test, label=name, return_scores=True)

                if debug:
                    print(f"\n👊 {name} {random.choice(child_comments)}")
                    print(f"📍 Style: {resampler_label}")
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
                    "lineage": lineage,
                    "dataset_name": dataset_name,         # ✅ Add here
                    "baseline_pr_auc": baseline_pr_auc    # ✅ Add here
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

            # ✅ Add generation table summary
            summary_table = []
            for model in newly_created:
                pr_auc = model["score"]["pr_auc"]
                summary_table.append({
                    "Generation": model["generation"],
                    "Dataset": dataset_name,
                    "Baseline PR AUC": round(baseline_pr_auc, 3),
                    "Label": f"{model['label']}_G{model['generation']}",
                    "PR AUC": round(pr_auc, 3),
                    "Beats Baseline?": "✅ Yes" if pr_auc > baseline_pr_auc else "❌ No",
                    "Lineage": model.get("lineage", "—"),
                    "Status": "✅ Survived" if id(model) in survivor_ids else "❌ Eliminated"
                })

            df_summary = pd.DataFrame(summary_table).sort_values(by=["Generation", "PR AUC"], ascending=[True, False])
            print(f"\n📋 Summary Table – Generation {gen} (Dataset: {dataset_name}):\n")
            print(df_summary)
            df_summary.to_csv(f"logs/generation_{gen}_{dataset_name}_summary.csv", index=False)

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
