import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import FullyConnected

def solve_matrix_multiplication_bound(a, b_l, b_u):
    """
    fast bound multiplication

    @params:
    a: layer weight
    b_l: lower bound for input
    b_u: upper bound for input

    @return:
    lower_bound, upper_bound
    """
    a_positive = F.relu(a)
    a_negative = -F.relu(-a)  # (100, 784)
    c_t_u = torch.matmul(a_positive, b_u) + torch.matmul(a_negative, b_l)
    c_t_l = torch.matmul(a_negative, b_u) + torch.matmul(a_positive, b_l)
    return c_t_l.transpose(0, 1), c_t_u.transpose(0, 1)


def solve_zonotope_bound(a_0, params, eps, need_relax):
    """
    Zonotope format: a_0 + \sum{params * eps}
    """
    if not need_relax:
        print(params.shape)
    pass


def get_network_real_input(network, image_matrix):
    if isinstance(network, FullyConnected):
        preprocess_layers = network.layers[:2]
    else:
        preprocess_layers = network.layers[:1]
    return preprocess_layers(image_matrix)


class Zonotope(object):
    def __init__(self, a_0, params, eps):
        self.a_0 = a_0
        self.params = params
        self.eps = eps

    def relax(self):
        # Passing Relu
        lower_bound, upper_bound = self.get_bound()
        print(lower_bound.shape)

    def get_bounds(self):
        eps_tensor = torch.Tensor(self.params.shape[1], 1)
        eps_tensor.fill_(self.eps)
        lower_bound, upper_bound = solve_matrix_multiplication_bound(self.params, -eps_tensor, eps_tensor)
        return lower_bound, upper_bound
