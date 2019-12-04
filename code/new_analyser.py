import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from networks import Normalization, FullyConnected, Conv
from new_utils import Zonotope, get_network_real_input

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
    layers = net.layers[2:]
    _ , first_fc_input_length  = inputs.shape
    eps_params = torch.Tensor(1, first_fc_input_length)
    eps_params.fill_(eps)
    z = Zonotope(inputs, eps_params, name="init")
    lower, upper = z.get_bound()
    # now enters fc-relu-fc
    i=0
    while(i < len(layers)):
        z = z.linear(layers[i].weight, layers[i].bias)
        i = i+1
        
        i = i+1

def analyse_conv(net, inputs, eps):
    pass
