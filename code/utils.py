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

def isVerified(lower, upper, real_label):
    true_lower = lower[real_label]
    true_upper = upper[real_label]
    upper.remove(true_upper)
    false_upper = max(upper)
    if (true_lower > false_upper):
        return True
    else:
        return False
