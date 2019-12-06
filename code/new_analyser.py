import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from networks import Normalization, FullyConnected, Conv
from new_utils import get_network_real_input, isLayerOutputCoveredbyBound
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
    layers = net.layers[2:]
    real_final_output = layers[0:3](inputs)
    _ , first_fc_input_length  = inputs.shape
    init_zonotopes = [Zonotope(a_0, [eps]) for a_0 in inputs[0]]
    previous_layer = Layer(init_zonotopes)
    i=0
    print(len(layers))
    while(i < len(layers)):
        # fc
        fc_layer = previous_layer
        fc_out = fc_layer.perform_linear(layers[i].weight, layers[i].bias)
        i = i+1
        print(isLayerOutputCoveredbyBound(fc_out, layers[:i], inputs))
        previous_layer = fc_out
        # relu
        if(i<len(layers)):
            relu_out = previous_layer.perform_relu()
            i = i+1
            print(isLayerOutputCoveredbyBound(relu_out, layers[:i], inputs))
            previous_layer = relu_out
    # now previous_layer becomes final output
    out = previous_layer
    lower, upper = out.calc_bounds()
    print(lower)
    print(real_final_output)
    print(upper)

def analyse_conv(net, inputs, eps):
    pass
