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
    # get bound for first fc layer
    a_0 = F.linear(inputs, net.layers[2].weight, net.layers[2].bias)
    a_0_params = net.layers[2].weight
    # now enters relu-fc-relu-fc...
    layers = net.layers[3:-1]
    print(layers)
    z = Zonotope(a_0, a_0_params, eps)      
    zonotopes = []
    for i in range(len(layers)):
        zonotopes = z.relax()
        i = i+1
    # now enters the final fc
    print(zonotopes)


def analyse_conv(net, inputs, eps):
    pass
