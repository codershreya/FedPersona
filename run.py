import os
import sys
import torch
import numpy as np
from data import get_cifar10, get_client_loaders, print_partition_stats
from client import FLClient
from train import run_fedavg, run_fedavg_ft, run_fedbn
from plot import (plot_learning_curves, plot_per_client_comparison,
                  plot_boxplot, print_summary_table)
from config import *

torch.manual_seed(SEED)
np.random.seed(SEED)

# Auto-detect Google Colab and mount Drive for persistent storage
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    PLOTS_DIR = "/content/drive/MyDrive/FL_Project/plots"
    import data as _data_module
    _data_module.DATA_ROOT = "/content/data"
else:
    PLOTS_DIR = "plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Colab: {IN_COLAB}")
    print(f"Config: {NUM_CLIENTS} clients | {NUM_ROUNDS} rounds | alpha={DIRICHLET_ALPHA}")
    print(f"Plots -> {PLOTS_DIR}")

    train_set, test_set = get_cifar10()
    train_loaders, test_loader, client_indices = get_client_loaders(train_set, test_set)

    print_partition_stats(client_indices, train_set)

    clients = [FLClient(i, train_loaders[i]) for i in range(NUM_CLIENTS)]

    # Experiment 1: FedAvg
    fedavg_round_accs, fedavg_client_accs, fedavg_model = run_fedavg(clients, test_loader)

    # Experiment 2: FedAvg + Fine-Tune (reuses trained FedAvg model)
    fedavg_ft_client_accs = run_fedavg_ft(clients, test_loader, pretrained_model=fedavg_model)

    # Experiment 3: FedBN (reset local BN states for a clean run)
    for c in clients:
        c._local_bn_state = None
    fedbn_round_accs, fedbn_client_accs, _ = run_fedbn(clients, test_loader)

    # Results
    print_summary_table(fedavg_client_accs, fedavg_ft_client_accs, fedbn_client_accs)

    plot_learning_curves(fedavg_round_accs, fedbn_round_accs,
                         save_path=f"{PLOTS_DIR}/learning_curves.png")
    plot_per_client_comparison(fedavg_client_accs, fedavg_ft_client_accs, fedbn_client_accs,
                                save_path=f"{PLOTS_DIR}/per_client_comparison.png")
    plot_boxplot(fedavg_client_accs, fedavg_ft_client_accs, fedbn_client_accs,
                 save_path=f"{PLOTS_DIR}/accuracy_boxplot.png")

    print(f"\nDone! Plots saved to {PLOTS_DIR}")
