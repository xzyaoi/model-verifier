import torch
import torch.nn as nn
import torch.nn.functional as F

from logger import logger

from utils import relu_relax_single_neuro


class Zonotope(object):

    def __init__(self, a_0, eps_params, name="null"):
        self.a_0 = a_0
        self.eps_params = eps_params
        self.name = name
        self.eps = 1

    def get_bound(self):
        self.eps_params = torch.Tensor(self.eps_params)
        positive = F.relu(self.eps_params)
        negative = -F.relu(-self.eps_params)
        upper = positive * self.eps + negative * (-self.eps)
        lower = positive * (-self.eps) + negative * (self.eps)
        return self.a_0 + sum(lower), self.a_0 + sum(upper)

    def linear(self, weight, bias):
        a_0 = F.linear(self.a_0, weight, bias)
        params = F.linear(self.eps_params, weight, bias)
        return Zonotope(a_0, params, "post_linear")


class Layer(object):
    def __init__(self, zonotopes):
        self.zonotopes = zonotopes

    def __len__(self):
        return sum([len(z.eps_params) for z in self.zonotopes])

    def perform_linear(self, weight, bias, after_relu=False):
        a_0 = torch.Tensor([z.a_0 for z in self.zonotopes])
        # shape is dim * k(#eps)
        if after_relu:
            original_params_map = torch.Tensor(
                [z.eps_params[:-1].detach().numpy() for z in self.zonotopes])
            extra_params_map = torch.diag(torch.Tensor(
                [z.eps_params[-1] for z in self.zonotopes]))
            # the last index of the error param before each affine layer is the new error term
            params_map = torch.cat([original_params_map, extra_params_map], 1)
        else:
            params_map = torch.diag(torch.flatten(torch.stack(
                [torch.Tensor(z.eps_params) for z in self.zonotopes])))
        new_a_0 = F.linear(a_0, weight, bias)
        new_params = F.linear(weight, torch.transpose(params_map, 0, 1))
        zonotopes = [Zonotope(a_0, eps)
                     for a_0, eps in zip(new_a_0, new_params)]
        return Layer(zonotopes)

    def perform_relu(self):
        previous_len = len(self)
        num_zonotopes = len(self.zonotopes)
        for index, item in enumerate(self.zonotopes):
            b_0, new_eps_params = relu_relax_single_neuro(
                item.a_0, item.eps_params)
            self.zonotopes[index] = Zonotope(b_0, new_eps_params)
        new_layer = Layer(self.zonotopes)
        # calc expected length
        expected_length = previous_len + num_zonotopes
        assert len(new_layer) == expected_length, "Size Mismatch, expected: %d != get:%d" % (expected_length, len(new_layer))
        return new_layer

    def calc_bounds(self):
        lowers = []
        uppers = []
        for each in self.zonotopes:
            lower, upper = each.get_bound()
            lowers.append(lower.data)
            uppers.append(upper.data)
        return lowers, uppers
