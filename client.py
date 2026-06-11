import copy
import torch
import torch.nn as nn
import torch.optim as optim
from config import *


class FLClient:
    def __init__(self, client_id, train_loader, device=DEVICE):
        self.id = client_id
        self.train_loader = train_loader
        self.device = device
        self._local_bn_state = None

    def _train_model(self, model, epochs, lr):
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()

        for _ in range(epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

        return model

    def train_fedavg(self, global_model, epochs=LOCAL_EPOCHS, lr=CLIENT_LR):
        local_model = copy.deepcopy(global_model).to(self.device)
        local_model = self._train_model(local_model, epochs, lr)
        return local_model.state_dict()

    def train_fedbn(self, global_model, bn_keys, epochs=LOCAL_EPOCHS, lr=CLIENT_LR):
        """
        FedBN: receive global non-BN params, restore local BN stats, train,
        cache updated BN stats, return only non-BN params to server.
        """
        local_model = copy.deepcopy(global_model).to(self.device)

        if self._local_bn_state is not None:
            sd = local_model.state_dict()
            for key in bn_keys:
                sd[key] = self._local_bn_state[key].to(self.device)
            local_model.load_state_dict(sd)

        local_model = self._train_model(local_model, epochs, lr)

        trained_sd = local_model.state_dict()
        self._local_bn_state = {k: v.clone().cpu()
                                 for k, v in trained_sd.items() if k in bn_keys}

        non_bn_sd = {k: v for k, v in trained_sd.items() if k not in bn_keys}
        return non_bn_sd

    def get_personalized_fedbn_model(self, global_model, bn_keys):
        """Reconstruct full personalized model: global non-BN params + local BN params."""
        model = copy.deepcopy(global_model).to(self.device)
        if self._local_bn_state is not None:
            sd = model.state_dict()
            for key in bn_keys:
                sd[key] = self._local_bn_state[key].to(self.device)
            model.load_state_dict(sd)
        return model

    def fine_tune(self, global_model, epochs=FINETUNE_EPOCHS, lr=CLIENT_LR * 0.1):
        local_model = copy.deepcopy(global_model).to(self.device)
        local_model = self._train_model(local_model, epochs, lr)
        return local_model

    @torch.no_grad()
    def evaluate(self, model):
        model = model.to(self.device)
        model.eval()
        correct, total = 0, 0
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
        return correct / total if total > 0 else 0.0

    def dataset_size(self):
        return len(self.train_loader.dataset)
