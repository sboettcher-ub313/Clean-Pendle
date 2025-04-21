import pandas as pd
import matplotlib.pyplot as plt

# 🏆 Print the winning model summary
def print_champion_summary(champion):
    print("\n🏆 GRAND CHAMPION 🏆")
    print(f"👑 Name: {champion['label']}_G{champion['generation']}")
    print(f"🔁 Generation: {champion['generation']}")
    print(f"🧬 Genome: C = {champion['params']['C']:.2f}, l1_ratio = {champion['params']['l1_ratio']:.2f}")
    print(f"📈 PR AUC: {champion['score']['pr_auc']:.3f}")
    print(f"🎖 ROC AUC: {champion['score']['roc_auc']:.3f}")
    print(f"🧬 Lineage: {champion.get('lineage', 'Origin Unknown')}")

# 📈 Plot evolution of PR AUC over generations
def plot_pr_auc_tracking(population):
    tracking_df = pd.DataFrame([
        {
            "Generation": model["generation"],
            "Strategy": model["label"],
            "PR AUC": model["score"]["pr_auc"],
            "ROC AUC": model["score"]["roc_auc"]
        }
        for model in population
    ])

    plt.figure(figsize=(10, 6))
    for strat in tracking_df["Strategy"].unique():
        group = tracking_df[tracking_df["Strategy"] == strat]
        plt.plot(group["Generation"], group["PR AUC"], marker="o", label=strat)

    plt.title("📈 PR AUC Evolution Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("PR AUC")
    plt.grid(True)
    plt.legend(loc="lower right", bbox_to_anchor=(1.0, 0))
    plt.tight_layout()
    plt.show()