import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class SlopeLoss(torch.nn.Module):
    def __init__(self):
        super(SlopeLoss, self).__init__()

    def forward(self, upper, lower, label):
        true_upper = upper.tolist()[label]
        true_lower = lower.tolist()[label]
        violate_u = [u for u in upper if u > true_lower]
        loss_val = sum(violate_u) / len(violate_u)
        loss_val.requires_grad = True
        print(loss_val)
        return loss_val


def verify(net, inputs, eps, label):
    # project the input
    upper = inputs + eps
    lower = inputs - eps
    diff = F.relu(upper - 1)
    upper = upper - diff
    lower = F.relu(lower)
    init_values = (upper + lower)
    eps = (init_values - lower)
    eps = eps.flatten().diag()
    init_eps = eps.view(list(init_values.shape) + [len(eps)])
    slopes = []
    is_init = True
    # go through network
    times = 10
    for i in range(times):
        relu_count = 0
        values = init_values.clone()
        eps = init_eps.clone()
        for layer in net.layers:
            if type(layer) is torch.nn.modules.linear.Linear:
                with torch.no_grad():
                    values, eps = affine_transform(values, eps, layer)
            elif type(layer) is torch.nn.modules.activation.ReLU:
                values, eps = relu_tranform(
                    values, eps, relu_count, slopes, is_init)
                relu_count += 1
            elif type(layer) is torch.nn.modules.Conv2d:
                values, eps = cnn_transform(values, eps, layer)
            elif type(layer) is torch.nn.modules.flatten.Flatten:
                # values = values.view(1,1, np.prod(values.shape),1)
                # eps = eps.view(1,1,np.prod(values.shape)).diag()
                values = values.flatten()  # 1 * nodes
                    # node_num * ep_num
                eps = eps.view(np.prod(values.shape), eps.shape[-1])
            else:
                # normalizaton
                mean = torch.FloatTensor([0.1307]).view((1, 1, 1, 1))
                sigma = torch.FloatTensor([0.3081]).view((1, 1, 1, 1))
                values = (values - mean) / sigma
                eps = eps / sigma
        is_init = False
        param_groups = []
        for each in slopes:
            param_groups.append({
                'params': each,
            })
        upper, lower = get_bound(values, eps)
        optimizer = torch.optim.SGD(param_groups, lr=0.2)
        # loss_func = nn.L1Loss()
        slopeloss = SlopeLoss()
        loss = slopeloss(upper, lower, label)
        optimizer.zero_grad()
        loss.backward()
        upper, lower = upper.tolist(), lower.tolist()
        upper.remove(upper[label])
        result = lower[label] > max(upper)
        if result:
            return result
        optimizer.step()
    return result


def affine_transform(values, eps, layer):
    # value: 1 * nodes
    # process values
    values = F.linear(values, layer.weight.detach(), layer.bias.detach())  # 1 * new_nodes
    # ep_num * nodes, weight: new_nodes * nodes
    eps = torch.matmul(layer.weight.detach(), eps)  # new_nodes, ep_num
    return values, eps


def relu_tranform(values, eps, relu_count, slopes, is_init):
    # eps: nodes * ep_num, values: 1 * nodes
    values_flat = values.flatten()
    # eps_flat = eps.view(np.prod(list(eps.shape)[:-1]), eps.shape[-1])
    # 1 * nodes
    eps = torch.cat((eps, torch.zeros(np.prod(values.shape), 1).view(
        list(eps.shape)[:-1] + [1])), dim=-1)
    eps_flat = eps.view(np.prod(list(eps.shape)[:-1]), eps.shape[-1])
    upper, lower = get_bound(values_flat, eps_flat)
    if is_init:
        current_slopes = torch.Tensor(upper/(upper-lower))
        print(current_slopes.shape)
        slopes.append(current_slopes)
    current_slopes = slopes[relu_count]
    for idx, _ in enumerate(values_flat):
        u, l = upper[idx], lower[idx]
        if u <= 0:
            values_flat[idx] = 0
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
            eps_flat[idx] = torch.cat(
                (slope * eps_flat[idx][:-1], torch.Tensor([term])))
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
                                     stride=list(layer.stride) + [1],
                                     padding=list(layer.padding) + [0])
    return values, eps
