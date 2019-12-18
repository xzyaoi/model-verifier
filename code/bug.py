import torch
import numpy as np
import torch.nn.functional as F


def verify(net, inputs, eps, label):
    # project the input
    upper = inputs + eps
    lower = inputs - eps
    diff = F.relu(upper - 1)
    upper = upper - diff
    lower = F.relu(lower)
    values = (upper + lower).detach()
    eps = (values - lower).detach()
    eps = eps.flatten().diag()
    eps = eps.view(list(values.shape) + [len(eps)])
    slopes = []
    relu_count = 0
    is_init = True
    # go through network
    for layer in net.layers:
        if type(layer) is torch.nn.modules.linear.Linear:
            values, eps = affine_transform(values, eps, layer)
        elif type(layer) is torch.nn.modules.activation.ReLU:
            values, eps = relu_tranform(values, eps, relu_count, slopes, is_init)
            relu_count += 1
        elif type(layer) is torch.nn.modules.Conv2d:
            values, eps = cnn_transform(values, eps, layer)
        elif type(layer) is torch.nn.modules.flatten.Flatten:
            # values = values.view(1,1, np.prod(values.shape),1)
            # eps = eps.view(1,1,np.prod(values.shape)).diag()  
            values = values.flatten()# 1 * nodes
            eps = eps.view(np.prod(values.shape), eps.shape[-1])# node_num * ep_num
        else:
            # normalizaton
            mean = torch.FloatTensor([0.1307]).view((1, 1, 1, 1))
            sigma = torch.FloatTensor([0.3081]).view((1, 1, 1, 1))
            values = (values - mean) / sigma
            eps = eps / sigma
    is_init = False
    upper, lower = get_bound(values, eps)
    # print(upper)
    # print(lower)
    slopeloss = SlopeLoss()
    loss = slopeloss(upper, lower, label)
    print(loss)
    loss.backward()
    upper, lower = upper.tolist(), lower.tolist()
    upper.remove(upper[label])
    return lower[label] > max(upper)


def affine_transform(values, eps, layer):
    # value: 1 * nodes
    # process values
    values = layer(values)  # 1 * new_nodes
    # ep_num * nodes, weight: new_nodes * nodes
    eps = torch.matmul(layer.weight, eps)  # new_nodes, ep_num
    return values, eps


def relu_tranform(values, eps, relu_count, slopes, is_init):
    # eps: nodes * ep_num, values: 1 * nodes
    values_flat = values.flatten()
    # eps_flat = eps.view(np.prod(list(eps.shape)[:-1]), eps.shape[-1])
    # 1 * nodes
    eps = torch.cat((eps, torch.zeros(np.prod(values.shape), 1).view(list(eps.shape)[:-1] + [1])), dim=-1)
    eps_flat = eps.view(np.prod(list(eps.shape)[:-1]), eps.shape[-1])
    upper, lower = get_bound(values_flat, eps_flat)
    if is_init:
        slopes.append(upper / (upper - lower))
    current_slopes = slopes[relu_count]
    for idx, _ in enumerate(values_flat):
        u, l = upper[idx], lower[idx]
        if u <= 0:
            values_flat[idx]=0
            eps_flat[idx].fill_(0)
        elif l >= 0:
            pass
        else:
            base = u / (u - l)
            # slope = np.random.uniform(0, 1)
            slope = current_slopes[idx]
            if slope <= base:
                term = (1 - slope) * u / 2
            else:
                term = -l * slope / 2
            values_flat[idx] = slope * values_flat[idx] + term
            eps_flat[idx] = torch.cat((slope * eps_flat[idx][:-1], torch.Tensor([term])))
    return values_flat.view(values.shape), eps_flat.view(eps.shape)

def get_bound(values, eps):
    abs_eps = torch.sum(torch.abs(eps), dim=-1)
    upper = values + abs_eps
    lower = values - abs_eps
    return upper, lower

def cnn_transform(values, eps, layer):
    values = layer(values)
    ep_shape = eps.shape
    eps = torch.nn.functional.conv3d(eps, layer.weight.view(list(layer.weight.shape) + [1]),
                                     stride = list(layer.stride) + [1],
                                     padding = list(layer.padding) + [0])
    return values, eps

class SlopeLoss(torch.nn.Module):
    def __init__(self):
        super(SlopeLoss, self).__init__()
    def forward(self, upper, lower, label):
        true_upper = upper.tolist()[label]
        true_lower = lower.tolist()[label]
        violate_u = [u for u in upper if u > true_lower]
        return sum(violate_u) / len(violate_u)


