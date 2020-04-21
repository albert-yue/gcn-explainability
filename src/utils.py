import torch
import torch.nn as nn
import torch.sparse as sparse

def get_trainable_parameters(model: nn.Module):
    return filter(lambda p: p.requires_grad, model.parameters())


def save_sparse_tensor(x, filepath):
    x = x.coalesce()
    x_dict = {'indices': x.indices(), 'values': x.values(), 'size': x.size()}
    torch.save(x_dict, filepath)


def load_sparse_tensor(filepath):
    x_dict = torch.load(filepath)
    return sparse.FloatTensor(x_dict['indices'], x_dict['values'], x_dict['size'])
