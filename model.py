import torch.nn as nn


class FedNet(nn.Module):
    """
    CNN for CIFAR-10 with BatchNorm after every conv/fc layer.
    BatchNorm layers are kept local in FedBN — they capture per-client data statistics.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def get_bn_keys(model):
    """Return state_dict keys belonging to BatchNorm layers (kept local in FedBN)."""
    bn_keys = set()
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            for suffix in ["weight", "bias", "running_mean",
                           "running_var", "num_batches_tracked"]:
                bn_keys.add(f"{name}.{suffix}")
    return bn_keys
