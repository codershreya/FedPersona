import torch

# Federation settings
NUM_CLIENTS       = 10
NUM_ROUNDS        = 50
LOCAL_EPOCHS      = 5
FINETUNE_EPOCHS   = 5
CLIENTS_PER_ROUND = 10

# Training settings
BATCH_SIZE   = 64
CLIENT_LR    = 0.01
MOMENTUM     = 0.9
WEIGHT_DECAY = 1e-4

# Non-IID control
# alpha=0.1  → very heterogeneous
# alpha=0.5  → moderately heterogeneous
# alpha=100  → near-IID
DIRICHLET_ALPHA = 0.3

NUM_CLASSES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
