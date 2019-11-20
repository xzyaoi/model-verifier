import numpy as np
from networks import Normalization, FullyConnected, Conv
import torch.nn as nn
import torch.nn.functional as F
import torch

DEVICE = 'cpu'


def fast_bound_multiplication(a, b_l, b_u):
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
    a_negative = -F.relu(-a)
    c_t_u = torch.matmul(a_positive, b_u) + torch.matmul(a_negative, b_l)
    c_t_l = torch.matmul(a_negative, b_u) + torch.matmul(a_positive, b_l)
    return c_t_l.transpose(0, 1), c_t_u.transpose(0, 1)


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
    c_l, c_u = fast_bound_multiplication(
        network.layers[4].weight, l_bounds_tensor.transpose(0, 1), u_bounds_tensor.transpose(0, 1))
    print(c_l)
    print(c_u)


def calc_bound_fc(layer, layer_input, eps):
    input_lower = (layer_input - eps).transpose(0, 1)
    input_upper = (layer_input + eps).transpose(0, 1)
    l, u = fast_bound_multiplication(
        layer.weight, input_lower, input_upper)
    return l, u


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
    # maybe boost by map
    for i in range(bound_length):
        slopes.append(get_slope(lower_bound[0, i], upper_bound[0, i]))
    for i in range(len(slopes)):
        l_bound, u_bound = calc_bounds(
            input_value[0, i], lower_bound[0, i], upper_bound[0, i], slopes[i])
        l_bounds.append(l_bound)
        u_bounds.append(u_bound)
    return l_bounds, u_bounds


def get_preprocess_result(network, image_matrix, eps):
    if isinstance(network, FullyConnected):
        preprocess_layers = network.layers[:2]
    else:
        preprocess_layers = network.layers[:1]
    return preprocess_layers(image_matrix)
