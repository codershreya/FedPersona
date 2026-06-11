import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

COLORS = {
    "FedAvg":    "#4C72B0",
    "FedAvg+FT": "#DD8452",
    "FedBN":     "#55A868",
}


def plot_learning_curves(fedavg_accs, fedbn_accs, save_path="plots/learning_curves.png"):
    plt.figure(figsize=(9, 5))
    rounds = range(1, len(fedavg_accs) + 1)
    plt.plot(rounds, fedavg_accs, label="FedAvg", color=COLORS["FedAvg"],  linewidth=2)
    plt.plot(rounds, fedbn_accs,  label="FedBN",  color=COLORS["FedBN"],   linewidth=2)
    plt.xlabel("Communication Round", fontsize=13)
    plt.ylabel("Global Test Accuracy", fontsize=13)
    plt.title(f"Global Test Accuracy over Rounds (α={_get_alpha()})", fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_per_client_comparison(fedavg_accs, fedavg_ft_accs, fedbn_accs,
                                save_path="plots/per_client_comparison.png"):
    n_clients = len(fedavg_accs)
    x = np.arange(n_clients)
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width, fedavg_accs,    width, label="FedAvg",    color=COLORS["FedAvg"],    alpha=0.85)
    ax.bar(x,         fedavg_ft_accs, width, label="FedAvg+FT", color=COLORS["FedAvg+FT"], alpha=0.85)
    ax.bar(x + width, fedbn_accs,     width, label="FedBN",     color=COLORS["FedBN"],     alpha=0.85)

    ax.set_xlabel("Client ID", fontsize=13)
    ax.set_ylabel("Local Accuracy", fontsize=13)
    ax.set_title(f"Per-Client Accuracy: FedAvg vs Personalized Methods (α={_get_alpha()})", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{i}" for i in range(n_clients)])
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.0)

    for name, accs, color in [
        ("FedAvg",    fedavg_accs,    COLORS["FedAvg"]),
        ("FedAvg+FT", fedavg_ft_accs, COLORS["FedAvg+FT"]),
        ("FedBN",     fedbn_accs,     COLORS["FedBN"]),
    ]:
        ax.axhline(np.mean(accs), color=color, linestyle="--", linewidth=1.2, alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_boxplot(fedavg_accs, fedavg_ft_accs, fedbn_accs,
                 save_path="plots/accuracy_boxplot.png"):
    fig, ax = plt.subplots(figsize=(8, 6))
    data = [fedavg_accs, fedavg_ft_accs, fedbn_accs]
    labels = ["FedAvg", "FedAvg+FT", "FedBN"]
    colors = [COLORS[l] for l in labels]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_ylabel("Per-Client Local Accuracy", fontsize=13)
    ax.set_title(f"Distribution of Per-Client Accuracy (α={_get_alpha()})", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_alpha_sensitivity(results_by_alpha, save_path="plots/alpha_sensitivity.png"):
    """
    results_by_alpha = {alpha: {"FedAvg": [...], "FedBN": [...], "FedAvg+FT": [...]}}
    """
    alphas = sorted(results_by_alpha.keys())
    fig, ax = plt.subplots(figsize=(9, 5))

    for method, color in COLORS.items():
        means = [np.mean(results_by_alpha[a][method]) for a in alphas]
        ax.plot(alphas, means, label=method, color=color, marker='o', linewidth=2)

    ax.set_xscale("log")
    ax.set_xlabel("Dirichlet α (log scale)", fontsize=13)
    ax.set_ylabel("Mean Per-Client Accuracy", fontsize=13)
    ax.set_title("Effect of Data Heterogeneity on Personalization", fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def print_summary_table(fedavg_accs, fedavg_ft_accs, fedbn_accs):
    print("\n" + "="*55)
    print(f"{'Method':<15} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-"*55)
    for name, accs in [
        ("FedAvg",    fedavg_accs),
        ("FedAvg+FT", fedavg_ft_accs),
        ("FedBN",     fedbn_accs),
    ]:
        print(f"{name:<15} {np.mean(accs):>8.4f} {np.std(accs):>8.4f} "
              f"{np.min(accs):>8.4f} {np.max(accs):>8.4f}")
    print("="*55)


def _get_alpha():
    from config import DIRICHLET_ALPHA
    return DIRICHLET_ALPHA
