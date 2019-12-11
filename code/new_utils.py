import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import FullyConnected


def get_network_real_input(network, image_matrix):
    if isinstance(network, FullyConnected):
        preprocess_layers = network.layers[:2]
    else:
        preprocess_layers = network.layers[:1]
    return preprocess_layers(image_matrix)

# Relu Relaxation methods


def relu_relax_single_neuro(a_0, eps_params):
    lower = 0
    upper = 0
    for each in eps_params:
        if (each > 0):
            lower = lower - each
            upper = upper + each
        else:
            upper = upper - each
            lower = lower + each
    l = a_0 + lower
    u = a_0 + upper
    if (u <= 0):
        new_eps_param = torch.cat((eps_params, torch.Tensor([0])))
        return 0, new_eps_param
    elif (l >= 0):
        new_eps_param = torch.cat((eps_params, torch.Tensor([0])))
        return a_0, new_eps_param
    else:
        slope = upper/(upper-lower)
        new_eps_param = torch.cat((eps_params, torch.Tensor([-slope*lower/2])))
        return slope*a_0-slope*lower/2, new_eps_param

def isLayerOutputCoveredbyBound(zono_layer, layers, inputs):
    reals = layers(inputs)
    lowers, uppers = zono_layer.calc_bounds()
    for i, item in enumerate(reals):
        l = lowers[i]
        u = uppers[i]
        real = reals[0, i]
        if not (l<real and u > real):
            print("[err] layer: " + str(layers[-1])+ "cannot pass boundary check!")
            return False
    print("[info] layer: " + str(layers[-1])+ "passed boundary check!")
    return True

def isVerified(lower, upper, real_label):
    true_lower = lower[real_label]
    true_upper = upper[real_label]
    print(true_upper)
    upper.remove(true_upper)
    print(true_lower)
    print(upper)
    false_upper = max(upper)
    print(false_upper)
    if (true_lower > false_upper):
        return True
    else:
        return False
