import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import FullyConnected


def get_network_real_input(network, image_matrix):
    if isinstance(network, FullyConnected):
        preprocess_layers = network.layers[:2]
    else:
        preprocess_layers = network.layers[:1]
    return preprocess_layers(image_matrix)

# Relu Relaxation methods


class Zonotope(object):

    def __init__(self, a_0, eps_params, name="null"):
        self.a_0 = a_0
        self.eps_params = eps_params
        self.name = name
        self.eps = 1
        assert self.a_0.shape == self.eps_params.shape, "Size mismatch for a_0 and eps params"

    def get_bound(self):
        positive = F.relu(self.eps_params)
        negative = -F.relu(-self.eps_params)
        upper = positive * self.eps + negative * (-self.eps)
        lower = positive * (-self.eps) + negative * (self.eps)
        return self.a_0 + lower, self.a_0 + upper

    def relax(self, method):
        if method == "relu":
            lower, upper = self.get_bound()
            _, bound_shape = lower.shape
        else:
            print("method not supported!")

    def linear(self, weight, bias):
        a_0 = F.linear(self.a_0, weight, bias)
        params = F.linear(self.eps_params, weight, bias)
        return Zonotope(a_0, params, "post_linear")
