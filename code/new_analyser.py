import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

from networks import Normalization, FullyConnected, Conv
from utils import get_network_real_input, \
    solve_matrix_multiplication_bound, \
    Zonotope

DEVICE = 'cpu'


def analyse_net(net, inputs, eps, true_label):
    real_network_input = get_network_real_input(net, inputs)
    if isinstance(net, FullyConnected):
        analyse_fc(net, real_network_input, eps)
    else:
        analyse_conv(net, real_network_input, eps)


def analyse_fc(net, inputs, eps):
    """
    network: fc-relu-fc-relu-fc
    """
    real_output = net.layers[2](inputs)
    a_0 = F.linear(inputs, net.layers[2].weight, net.layers[2].bias)
    a_0_params = net.layers[2].weight
    # now enters relu-fc-relu-fc...
    layers = net.layers[3:]
    # initial zonotope
    z = Zonotope(a_0, a_0_params, eps, name="initial_zonotopes")
    i=0
    while(i<len(layers)-1):
        # relu
        # z.print()
        print(net.layers[:3+i])
        real_relu_input = net.layers[:3+i](inputs)
        z = z.relax(real_relu_input)
        z.print()
        lower_bound, upper_bound = z.get_bounds()
        print(lower_bound)
        print(real_output)
        print(upper_bound)
        # fc
        i=i+1
        z = z.linear(layers[i].weight, layers[i].bias)
        i=i+1
    lower_bounds, upper_bounds = z.get_final_bounds()
    # now enters the final fc

def analyse_fc2(net, input, eps):
    pass

def analyse_conv(net, inputs, eps):
    pass
