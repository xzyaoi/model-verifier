import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import FullyConnected
from logger import logger

def get_network_real_input(network, image_matrix):
    if isinstance(network, FullyConnected):
        preprocess_layers = network.layers[:2]
    else:
        preprocess_layers = network.layers[:1]
    return preprocess_layers(image_matrix)

def isLayerOutputCoveredbyBound(zono_layer, layers, inputs):
    reals = layers(inputs)
    lowers, uppers = zono_layer.calc_bounds()
    for i, item in enumerate(reals):
        l = lowers[i]
        u = uppers[i]
        real = reals[0, i]
        if not (l<real and u > real):
            print('error')
            logger.error("[err] layer: " + str(layers[-1])+ " cannot pass boundary check!")
            return False
    logger.info("[info] layer: " + str(layers[-1])+ " passed boundary check!")
    return True

def isVerified(lower, upper, real_label):
    true_lower = lower[real_label]
    true_upper = upper[real_label]
    upper.remove(true_upper)
    false_upper = max(upper)
    if (true_lower > false_upper):
        return True
    else:
        return False
