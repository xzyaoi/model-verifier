import numpy as np
from networks import Normalization, FullyConnected, Conv
import torch.nn as nn

DEVICE = 'cpu'


def analyse_fc(network, image_matrix, eps):
    # get input for fc
    l_fc_input_0, r_fc_input_0, u_fc_input_0 = get_preprocess_result(
        network, image_matrix, eps)
    # next: fc-relu-fc-relu-fc...(will end with fc)
    fc_relu_layers = network.layers[2:-1]
    for each in fc_relu_layers:
        if isinstance(each, nn.Linear):
            # there should be 100 neurons
            # access to weights and bias: each.weight, each.bias
            a_0_s = np.dot(r_fc_input_0.detach().numpy(), each.weight.detach().numpy().transpose())
            epsilons = np.dot(eps, each.weight.detach().numpy().transpose())
            print(epsilons)
            print(epsilons.shape)
            # verify_relu(a_0_s, epsilons)
        else:
            pass


def verify_relu(lower_bound, upper_bound, real):
    # we need it to be zonotope format
    def get_output(l, u, r):
        if (u <= 0):
            return 0
        elif (l >= 0):
            return r
        else:
            slope = u/(u-l)
            # Zonotope steps 3 in Lecture 4: Verify


def get_preprocess_result(network, image_matrix, eps):
    lower_bound_image_matrix = image_matrix - eps
    upper_bound_image_matrix = image_matrix + eps
    if isinstance(network, FullyConnected):
        preprocess_layers = network.layers[:2]
    else:
        preprocess_layers = network.layers[:1]
    return preprocess_layers(lower_bound_image_matrix),\
            preprocess_layers(image_matrix), \
            preprocess_layers(upper_bound_image_matrix)
