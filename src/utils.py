import torch
import torch.nn as nn

def get_trainable_parameters(model: nn.Module):
    return filter(lambda p: p.requires_grad, model.parameters())
