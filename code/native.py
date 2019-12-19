import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Zonotope(nn.Module):
    def __init__(self, layers, eps):
        super(Zonotope, self).__init__()
        self.layers = layers
        self.slopes = []
        self.eps = eps

    def affine(self, values, eps, layer):
        return F.linear(values, layer.weight, layer.bias), torch.matmul(layer.weight, eps)
    
    def conv(self, values, eps, layer):
        values = layer(values)
        ep_shape = eps.shape
        eps = torch.nn.functional.conv3d(eps, layer.weight.view(list(layer.weight.shape) + [1]),
                                        stride=list(layer.stride) + [1],
                                        padding=list(layer.padding) + [0])
        return values, eps
    
    def relu(self, values, eps, relu_count):
        values_flat = values.flatten()
        # eps_flat = eps.view(np.prod(list(eps.shape)[:-1]), eps.shape[-1])
        # 1 * nodes
        eps = torch.cat((eps, torch.zeros(np.prod(values.shape), 1).view(
            list(eps.shape)[:-1] + [1])), dim=-1)
        eps_flat = eps.view(np.prod(list(eps.shape)[:-1]), eps.shape[-1])
        upper, lower = self.get_bound(values_flat, eps_flat)
        if len(self.slopes)<relu_count+1:
            # init slopes
            current_slopes = torch.nn.Parameter(upper/(upper-lower))
            self.slopes.append(current_slopes)
        current_slopes = self.slopes[relu_count]
        for idx, _ in enumerate(values_flat):
            u, l = upper[idx], lower[idx]
            if u <= 0:
                values_flat[idx] = 0
                eps_flat[idx].fill_(0)
                pass
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
                new_val = slope.data * values_flat[idx].data
                values_flat[idx] = new_val + term
                eps_flat[idx] = torch.cat(
                    (slope * eps_flat[idx][:-1], torch.Tensor([term])))
        return values_flat.view(values.shape), eps_flat.view(eps.shape)


    def get_bound(self, values, eps):
        abs_eps = torch.abs(eps.detach())
        sum_eps = torch.sum(abs_eps, dim=-1)
        # abs_eps = torch.sum(torch.abs(eps), dim=-1)
        upper = values + sum_eps
        lower = values - sum_eps
        return upper, lower

    def forward(self, inputs):
        # preprocess
        eps = self.eps
        upper = inputs + eps
        lower = inputs - eps
        diff = F.relu(upper - 1)
        upper = upper - diff
        lower = F.relu(lower)
        values = (upper + lower)
        eps = (values - lower)
        eps = eps.flatten().diag()
        relu_count = 0
        # real forward
        for layer in self.layers:
            if type(layer) is torch.nn.modules.linear.Linear:
                values, eps = self.affine(values, eps, layer)
            elif type(layer) is torch.nn.modules.activation.ReLU:
                values, eps = self.relu(
                    values, eps, relu_count)
                relu_count += 1
            elif type(layer) is torch.nn.modules.Conv2d:
                values, eps = self.conv(values, eps, layer)
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
        upper, lower = self.get_bound(values, eps)
        return upper, lower

class SlopeLoss(torch.nn.Module):
    def __init__(self):
        super(SlopeLoss, self).__init__()

    def forward(self, upper, lower, label):
        true_upper = upper.tolist()[label]
        true_lower = lower.tolist()[label]
        violate_u = [u-true_lower for u in upper if u > true_lower]
        loss_val = sum(violate_u) / len(violate_u)
        return loss_val, len(violate_u)