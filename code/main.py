import torch
import numpy as np
from torch.nn.functional import relu
from networks import FullyConnected

def get_network_real_input(network, image_matrix, eps):
    x = image_matrix + eps
    x = x - relu(x - 1)
    y = relu(image_matrix - eps)
    new_image_input = 1/2 * (x + y)
    print(new_image_input)
    new_eps = 1/2* (x+y) -y
    if isinstance(network, FullyConnected):
        preprocess_layers = network.layers[:2]
    else:
        preprocess_layers = network.layers[:1]
    return new_eps, preprocess_layers(image_matrix)

def verify_fc(net, a_0, eps_params, true_label):
    print(a_0)
    print(eps_params)
    layers = net.layers[2:]
    i = 0
    while(i < len(layers)):
        new_a0 = layers[i](a_0)
        

def verify_cnn(net, a_0, eps_params, true_label):
    pass

def verify(net, inputs, eps, true_label):
    eps_params, a_0 = get_network_real_input(net, inputs, eps)
    if isinstance(net, FullyConnected):
        return verify_fc(net, a_0, eps_params, true_label)
    else:
        return verify_cn(net, a_0, eps_params, true_label)