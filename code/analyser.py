import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from networks import Normalization, FullyConnected, Conv
from utils import isLayerOutputCoveredbyBound, isVerified
# from zonotope import Zonotope, Layer
from new_zonotope import ZonoFullyConnected, Zonotope, Layer
from main import get_network_real_input
DEVICE = 'cpu'

def analyse_net(net, inputs, eps, true_label):
    print("Epsilon: %f" % eps)
    print("True Label: %d" % true_label)
    eps, real_network_input = get_network_real_input(net, inputs, eps)
    if isinstance(net, FullyConnected):
        return analyse_fc(net, real_network_input, eps, true_label)
    else:
        return analyse_conv(net, real_network_input, eps, true_label)


def analyse_fc(net, inputs, eps, true_label):
    """
    network: fc-relu-fc-relu-fc
    """
    layers = net.layers[2:]
    _, first_fc_input_length  = inputs.shape
    # init_zonotopes = [Zonotope(a_0, torch.Tensor([eps])) for a_0 in inputs[0]]
    print(eps.shape)
    print(inputs.shape)
    init_zonotopes = [Zonotope(a_0, [eps]) for a_0, eps in zip(inputs, eps)]
    previous_layer = Layer(init_zonotopes)

    i=0
    while(i < len(layers)):
        # fc
        fc_layer = previous_layer.toFC()
        fc_out = fc_layer.forward(layers[i].weight, layers[i].bias, i != 0)
        i = i+1
        # isLayerOutputCoveredbyBound(fc_out, layers[:i], inputs)
        previous_layer = fc_out
        # relu
        if(i<len(layers)):
            relu_out = previous_layer.activate_relu()
            i = i+1
            # isLayerOutputCoveredbyBound(relu_out, layers[:i], inputs)
            previous_layer = relu_out
    # now previous_layer becomes final output
    out = previous_layer
    lower, upper = out.calc_bounds()
    return isVerified(lower, upper, true_label)

def analyse_conv(net, inputs, eps, true_label):
    print(inputs.shape)
    layers = net.layers[1:]
    init_zonotopes = inputs.tolist()
    print(init_zonotopes)
    init_zonotopes = [Zonotope(a_0, torch.Tensor([eps])) for a_0 in inputs[0]]
    previous_layer = Layer(init_zonotopes)
    inp_unf = torch.nn.functional.unfold(inputs, (4, 5))
    # conv-relu-flatten-linear-relu-linear
    i=0
    while(i<len(layers)):
        # conv
        # print(layers[i].weight.shape)
        w = layers[i].weight
        new_a0 = layers[i]()
        out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
        i = i + 1
    return False

def analyse_conv_simple(net, inputs, eps, true_label):
    print(inputs.shape)