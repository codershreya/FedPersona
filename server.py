import copy
from config import *


def fedavg_aggregate(global_model, client_state_dicts, client_sizes):
    """Weighted average of all client state dicts by dataset size."""
    total_size = sum(client_sizes)
    weights = [s / total_size for s in client_sizes]

    new_sd = copy.deepcopy(global_model.state_dict())

    for key in new_sd:
        new_sd[key] = sum(
            w * sd[key].float()
            for w, sd in zip(weights, client_state_dicts)
        ).to(new_sd[key].dtype)

    new_model = copy.deepcopy(global_model)
    new_model.load_state_dict(new_sd)
    return new_model


def fedbn_aggregate(global_model, client_non_bn_dicts, client_sizes, bn_keys):
    """Weighted average of non-BN parameters only; BN params remain local."""
    total_size = sum(client_sizes)
    weights = [s / total_size for s in client_sizes]

    new_sd = copy.deepcopy(global_model.state_dict())

    for key in new_sd:
        if key in bn_keys:
            continue
        new_sd[key] = sum(
            w * sd[key].float()
            for w, sd in zip(weights, client_non_bn_dicts)
        ).to(new_sd[key].dtype)

    new_model = copy.deepcopy(global_model)
    new_model.load_state_dict(new_sd)
    return new_model
