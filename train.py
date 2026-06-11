import torch
import copy
import numpy as np
from model import FedNet, get_bn_keys
from client import FLClient
from server import fedavg_aggregate, fedbn_aggregate
from config import *


def run_fedavg(clients, test_loader):
    print("\n" + "="*50)
    print("Running FedAvg")
    print("="*50)

    global_model = FedNet(NUM_CLASSES).to(DEVICE)
    round_accs = []

    for round_num in range(1, NUM_ROUNDS + 1):
        client_sds, sizes = [], []
        for client in clients:
            sd = client.train_fedavg(global_model)
            client_sds.append(sd)
            sizes.append(client.dataset_size())

        global_model = fedavg_aggregate(global_model, client_sds, sizes)

        acc = evaluate_global(global_model, test_loader)
        round_accs.append(acc)

        if round_num % 10 == 0:
            print(f"  Round {round_num:3d}/{NUM_ROUNDS} | Global Test Acc: {acc:.4f}")

    per_client_accs = [c.evaluate(global_model) for c in clients]
    print(f"\nFedAvg    | Mean: {np.mean(per_client_accs):.4f} "
          f"| Std: {np.std(per_client_accs):.4f} "
          f"| Min: {np.min(per_client_accs):.4f}")

    return round_accs, per_client_accs, global_model


def run_fedavg_ft(clients, test_loader, pretrained_model=None):
    print("\n" + "="*50)
    print("Running FedAvg + Fine-Tune")
    print("="*50)

    if pretrained_model is not None:
        global_model = copy.deepcopy(pretrained_model)
        print("  (Using pretrained FedAvg model)")
    else:
        global_model = FedNet(NUM_CLASSES).to(DEVICE)
        for round_num in range(1, NUM_ROUNDS + 1):
            client_sds, sizes = [], []
            for client in clients:
                sd = client.train_fedavg(global_model)
                client_sds.append(sd)
                sizes.append(client.dataset_size())
            global_model = fedavg_aggregate(global_model, client_sds, sizes)
            if round_num % 10 == 0:
                acc = evaluate_global(global_model, test_loader)
                print(f"  Round {round_num:3d}/{NUM_ROUNDS} | Global Test Acc: {acc:.4f}")

    print("  Fine-tuning clients...")
    per_client_accs = []
    for client in clients:
        ft_model = client.fine_tune(global_model)
        acc = client.evaluate(ft_model)
        per_client_accs.append(acc)

    print(f"\nFedAvg+FT | Mean: {np.mean(per_client_accs):.4f} "
          f"| Std: {np.std(per_client_accs):.4f} "
          f"| Min: {np.min(per_client_accs):.4f}")

    return per_client_accs


def run_fedbn(clients, test_loader):
    print("\n" + "="*50)
    print("Running FedBN")
    print("="*50)

    global_model = FedNet(NUM_CLASSES).to(DEVICE)
    bn_keys = get_bn_keys(global_model)
    round_accs = []

    for round_num in range(1, NUM_ROUNDS + 1):
        client_non_bn_sds, sizes = [], []
        for client in clients:
            non_bn_sd = client.train_fedbn(global_model, bn_keys)
            client_non_bn_sds.append(non_bn_sd)
            sizes.append(client.dataset_size())

        global_model = fedbn_aggregate(global_model, client_non_bn_sds, sizes, bn_keys)

        acc = evaluate_global(global_model, test_loader)
        round_accs.append(acc)

        if round_num % 10 == 0:
            print(f"  Round {round_num:3d}/{NUM_ROUNDS} | Global Test Acc: {acc:.4f}")

    per_client_accs = []
    for client in clients:
        personalized = client.get_personalized_fedbn_model(global_model, bn_keys)
        acc = client.evaluate(personalized)
        per_client_accs.append(acc)

    print(f"\nFedBN     | Mean: {np.mean(per_client_accs):.4f} "
          f"| Std: {np.std(per_client_accs):.4f} "
          f"| Min: {np.min(per_client_accs):.4f}")

    return round_accs, per_client_accs, global_model


@torch.no_grad()
def evaluate_global(model, test_loader):
    model.eval()
    correct, total = 0, 0
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += len(y)
    return correct / total
