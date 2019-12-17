import torch
import numpy as np
import torch.nn.functional as F
from networks import FullyConnected, Normalization
import torch.nn as nn

sigma = torch.FloatTensor([0.3081]).view((1, 1, 1, 1))
mean = torch.FloatTensor([0.1307]).view((1, 1, 1, 1))

def get_network_real_input(network, image_matrix, eps):
    x = image_matrix + eps
    x = x - F.relu(x - 1)
    y = F.relu(image_matrix - eps)
    new_image_input = 1/2 * (x + y)
    new_eps = 1/2 * (x+y) - y
    # print("New Epsilon: %f" % new_eps[0][0][0][0])
    if isinstance(network, FullyConnected):
        preprocess_layers = network.layers[:2]
    else:
        preprocess_layers = network.layers[:1] 
    # return new_eps, preprocess_layers(new_image_input)
    return new_eps, preprocess_layers(new_image_input)


def get_bounds(a_0, eps_params):
    positive = F.relu(eps_params)
    negative = -F.relu(-eps_params)
    upper = positive - negative
    lower = negative - positive
    # result is [1, 100]
    return a_0+torch.sum(lower, dim=0), a_0 + torch.sum(upper, dim=0)


def relu_relax(a_0, eps_params):
    lower, upper = get_bounds(a_0, eps_params)
    length, width = eps_params.shape
    new_eps_params = torch.Tensor(length+1, width)
    for index, item in enumerate(a_0[0]):
        l = lower[0, index]
        u = upper[0, index]
        if (u <= 0):
            a_0[0, index] = 0
            new_eps_params[:, index].fill_(0.0)
            new_eps_param = torch.Tensor([0.0])
            slope = 1
        elif (l >= 0):
            new_eps_param = torch.Tensor([0.0])
            slope = 1
        else:
            # a_0[0, index] = u / 2
            # new_eps_params[:, index].fill_(0.0)
            # new_eps_param = torch.Tensor([u/2])
            # # cross boundary
            slope = u/(u-l)
            a_0[0, index] = slope * a_0[0, index] - slope * l / 2
            new_eps_param = torch.Tensor([-slope*l/2])
        
        new_eps_params[:, index] = torch.cat(
            (eps_params[:, index] * slope, new_eps_param))
    return a_0, new_eps_params


def verify_fc(net, a_0, eps_params, true_label):
    layers = net.layers[2:]
    i = 0
    while(i < len(layers)):
        a_0 = layers[i](a_0)
        if i==0:
            eps_params = eps_params.diag()
        eps_params = F.linear(eps_params, layers[i].weight)
        i = i + 1
        # relu
        if(i < len(layers)):
            a_0, eps_params = relu_relax(a_0, eps_params)
            i = i + 1
    # output
    l, u = get_bounds(a_0, eps_params)
    return l, u


def verify_cnn(net, a_0, eps_params, true_label):
    print(a_0.shape)
    layers = net.layers[1:]
    i = 0
    while(i<len(layers)):
        print(layers[i])
        # conv
        a_0 = layers[i](a_0)
        
        eps_params = layers[i](a_0)
        i = i + 1


def verify(net, inputs, eps, true_label):
    eps_params, a_0 = get_network_real_input(net, inputs, eps)
    eps_params = eps_params / sigma
    eps_params = eps_params.flatten()
    if isinstance(net, FullyConnected):
        l, u = verify_fc(net, a_0, eps_params, true_label)
        print(l)
        print(u)
        return determine(l.tolist()[0], u.tolist()[0])
    else:
        return verify_cnn(net, a_0, eps_params, true_label)


def determine(l, u):
    u.remove(max(u))
    return max(l) >= max(u)