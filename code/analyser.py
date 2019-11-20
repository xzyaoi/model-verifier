import numpy as np
from networks import Normalization, FullyConnected, Conv
import torch.nn as nn
import torch

DEVICE = 'cpu'


def bound_multiplication(a, b, eps):
    """
    a is weight
    b is input, where perturbation happens
    """
    ar, ac = a.shape
    br, bc = b.shape
    assert (ac == br), "Shape Mismatch! %d != %d " % (ac, br)
    c_l = torch.zeros(ar, bc)
    c_u = torch.zeros(ar, bc)
    # TODO: use einsum or map at least to boost!!
    for i in range(ar):
        for j in range(bc):
            for k in range(ac):
                if a[i, k] >= 0:
                    c_u[i, j] += a[i, k] * (b[k, j] + eps)
                    c_l[i, j] += a[i, k] * (b[k, j] - eps)
                else:
                    c_u[i, j] += a[i, k] * (b[k, j] - eps)
                    c_l[i, j] += a[i, k] * (b[k, j] + eps)
    return c_l.transpose(0, 1), c_u.transpose(0, 1)

def bound_multiplication_2(a, b_u, b_l):
    """
    a is weight
    b is input, where perturbation happens
    """
    ar, ac = a.shape
    br, bc = b_u.shape
    assert (ac == br), "Shape Mismatch! %d != %d " % (ac, br)
    c_l = torch.zeros(ar, bc)
    c_u = torch.zeros(ar, bc)
    # TODO: use einsum or map at least to boost!!
    for i in range(ar):
        for j in range(bc):
            for k in range(ac):
                if a[i, k] >= 0:
                    c_u[i, j] += a[i, k] * (b_u[k,j])
                    c_l[i, j] += a[i, k] * (b_l[k,j])
                else:
                    c_u[i, j] += a[i, k] * (b_l[k,j])
                    c_l[i, j] += a[i, k] * (b_u[k,j])
    return c_l.transpose(0, 1), c_u.transpose(0, 1)

"""
def fast_bound_multiplication(a,b,eps):
    
    # a is weight
    # b is input, where perturbation happens
    
    b_u = b + eps
    b_l = b - eps
    c_t_u = torch.einsum('ik, kj -> ij', a, b_u)
    c_t_l = torch.einsum('ik, kj -> ij', a, b_l)
"""


def analyse_fc(network, image_matrix, eps_init):
    fc_input = get_preprocess_result(
        network, image_matrix, eps_init)
    first_fc_layer = network.layers[2]
    # a_0_s = torch.matmul(fc_input, first_fc_layer.weight.transpose(0,1)) + first_fc_layer.bias
    a_0_l, a_0_u = calc_bound_fc(first_fc_layer, fc_input, eps_init)
    l_bounds, u_bounds = calc_bound_relu(
        a_0_l, a_0_u, first_fc_layer(fc_input))
    # next fc layer is a matrix multiplication
    l_bounds_tensor = torch.Tensor([l_bounds])
    u_bounds_tensor = torch.Tensor([u_bounds])
    print(network.layers[4].weight.shape)
    print(l_bounds_tensor.shape)
    c_l, c_u = bound_multiplication_2(network.layers[4].weight, u_bounds_tensor.transpose(0,1), l_bounds_tensor.transpose(0,1))
    print(c_l)
    print(c_u)

def calc_bound_fc(layer, layer_input, eps):
    return bound_multiplication(layer.weight, layer_input.transpose(0, 1), eps)


def calc_bound_relu(lower_bound, upper_bound, input_value):
    bounds_coeffis = []
    slopes = []
    u_bounds = []
    l_bounds = []

    def get_slope(l, u):
        return u/(u-l)

    def calc_bounds(a0, l, u, slope):
        if (u <= 0):
            return 0, 0
        elif (l > 0):
            return a0.data, a0.data
        else:
            coeffi = (slope * l / 2)
            irr_effi = slope*a0 - slope*l/2
            if (coeffi >= 0):
                # return lower_bound, upper_bound
                return (irr_effi - coeffi).data, (irr_effi + coeffi).data
            else:
                return (irr_effi + coeffi).data, (irr_effi - coeffi).data

    # Calculate the slope for each error term
    _, bound_length = lower_bound.shape
    for i in range(bound_length):
        slopes.append(get_slope(lower_bound[0, i], upper_bound[0, i]))
    for i in range(len(slopes)):
        l_bound, u_bound = calc_bounds(
            input_value[0,i], lower_bound[0, i], upper_bound[0, i], slopes[i])
        l_bounds.append(l_bound)
        u_bounds.append(u_bound)
    return l_bounds, u_bounds


def get_preprocess_result(network, image_matrix, eps):
    if isinstance(network, FullyConnected):
        preprocess_layers = network.layers[:2]
    else:
        preprocess_layers = network.layers[:1]
    return preprocess_layers(image_matrix)
