import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from networks import Normalization, FullyConnected, Conv
from new_utils import get_network_real_input
from zonotope import Zonotope, Layer

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
    init_zonotopes = [Zonotope(a_0, [eps]) for a_0 in inputs[0]]
    previous_layer = Layer(init_zonotopes)
    i=0
    print(len(layers))
    while(i < len(layers) - 2):
        # fc
        fc_layer = previous_layer
        fc_out = fc_layer.perform_linear(layers[i].weight, layers[i].bias)
        lower1, upper1 = fc_out.calc_bounds()
        print(lower1)
        print(torch.Tensor(lower1))
        print(real_output)
        print(torch.Tensor(upper1))
        i = i+1
        # relu
        relu_out = fc_out.perform_relu()
        lower, upper = relu_out.calc_bounds()
        print(lower)
        print(real_output)
        print(upper)
        previous_layer = relu_out
        # print(len(relu_out))
        # print(lower)
        # print(real_output)
        # print(upper)
        i = i+1

def analyse_conv(net, inputs, eps):
    pass
