import torch
import numpy as np
import torch.nn.functional as F


def verify(net, inputs, eps):
    # project the input
    upper = inputs + eps
    lower = inputs - eps
    diff = F.relu(upper - 1)
    upper = upper - diff
    lower = F.relu(lower)
    values = (upper + lower)
    eps = (values - lower)
    eps = eps.view(list(values.shape) + [1])
    # go through network
    for layer in net.layers:
        if type(layer) is torch.nn.modules.linear.Linear:
            values, eps = affine_transform(values, eps, layer)
        elif type(layer) is torch.nn.modules.activation.ReLU:
            values, eps = relu_tranform(values, eps)
        elif type(layer) is torch.nn.modules.Conv2d:
            values, eps = cnn_transform(values, eps)
        elif type(layer) is torch.nn.modules.flatten.Flatten:
            # values = values.view(1,1, np.prod(values.shape),1)
            # eps = eps.view(1,1,np.prod(values.shape)).diag()  
            values = values.flatten()# 1 * nodes
            eps = eps.view(np.prod(values.shape), eps.shape[-1])# node_num * ep_num
            eps = eps.diag()
        else:
            # normalizaton
            mean = torch.FloatTensor([0.1307]).view((1, 1, 1, 1))
            sigma = torch.FloatTensor([0.3081]).view((1, 1, 1, 1))
            values = (values - mean) / sigma
            eps = eps / sigma
    upper, lower = get_bound(values, eps)
    print(upper)
    print(lower)
    upper, lower = upper.tolist(), lower.tolist()
    upper.remove(max(upper))
    return max(lower) > max(upper)


def affine_transform(values, eps, layer):
    # value: 1 * nodes
    # process values
    values = layer(values)  # 1 * new_nodes
    # ep_num * nodes, weight: new_nodes * nodes
    eps = torch.matmul(layer.weight, eps)  # new_nodes, ep_num
    return values, eps


def relu_tranform(values, eps):
    # eps: nodes * ep_num, values: 1 * nodes
    upper, lower = get_bound(values, eps)
    # 1 * nodes
    eps = torch.cat((eps, torch.zeros(len(values), 1)), dim=1)
    for idx, _ in enumerate(values):
        u, l = upper[idx], lower[idx]
        if u <= 0:
            values[idx]=0
            eps[idx].fill_(0)
        elif l >= 0:
            pass
        else:
            base = u / (u - l)
            # slope = np.random.uniform(0, 1)
            slope = 0.01
            if slope <= base:
                term = (1 - slope) * u / 2
            else:
                term = -l * slope / 2
            values[idx] = slope * values[idx] + term
            eps[idx] = torch.cat((slope * eps[idx][:-1], torch.Tensor([term])))
    return values, eps

def get_bound(values, eps):
    abs_eps = torch.sum(torch.abs(eps), dim=-1)
    upper = values + abs_eps
    lower = values - abs_eps
    return upper, lower

def cnn_transform(values, eps, layer):
    values = layer(values)
    ep_shape = eps.shape
    eps = torch.nn.functional.conv3d(values.view(eps, layer.weight.view(list(layer.weight.shape) + [1])))
    return values, eps
