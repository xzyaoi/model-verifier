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


def solve_matrix_multiplication_bound(a, b_l, b_u):
    """
    fast bound multiplication

    @params:
    a: layer weight
    b_l: lower bound for input
    b_u: upper bound for input

    @return:
    lower_bound, upper_bound
    """
    a_positive = F.relu(a)
    a_negative = -F.relu(-a)  # (100, 784)
    c_t_u = torch.matmul(a_positive, b_u) + torch.matmul(a_negative, b_l)
    c_t_l = torch.matmul(a_negative, b_u) + torch.matmul(a_positive, b_l)
    return c_t_l.transpose(0, 1), c_t_u.transpose(0, 1)

# =======
# Relu Zonotope Relaxation
# =======


def relu_relaxation(input_a0, lower_bound, upper_bound):
    def calc_each_bounds(a_0, l, u):
        if (u <= 0):
            return 0
        elif (l >= 0):
            return a_0
        else:
            slope = u/(u-l)
            return slope * a_0 - slope * l / 2, slope * l / 2, -1
    _, bound_length = lower_bound.shape
    # we will have #bound_length zonotopes
    zonotopes = []
    for i in range(bound_length):
        new_a0, new_params, new_eps = calc_each_bounds(
            input_a0[0, i], lower_bound[0, i], upper_bound[0, i])
        zonotopes.append(Zonotope(new_a0, new_params, new_eps))
    return zonotopes


class Zonotope(object):
    def __init__(self, a_0, params, eps):
        self.a_0 = a_0
        self.params = params
        self.eps = eps

    def relax(self, method='relu'):
        # Passing Relu
        zonotopes = []
        lower_bound, upper_bound = self.get_bounds()
        if method == 'relu':
            zonotopes = relu_relaxation(self.a_0, lower_bound, upper_bound)
        else:
            pass
        return zonotopes
    
    def get_bounds(self):
        eps_tensor = torch.Tensor(self.params.shape[1], 1)
        eps_tensor.fill_(self.eps)
        lower_bound, upper_bound = solve_matrix_multiplication_bound(
            self.params, -eps_tensor, eps_tensor)
        return lower_bound, upper_bound

class Zono(object):
    #define by values, and the params of the first layer inputs
    def __init__(self, u_value, u_params, l_params, l_value):
        self.u_value = u_value
        # the params base on the previous layer
        self.u_params = u_params
        self.l_value = l_value
        self.l_params = l_params
        self.previous
        
    def linear_relax(self, weights, bias, pre_zonos):
        new_u_params = F.linear(self.u_params, weight, bias)
        new_l_params = F.linear(self.l_params, weight, bias)
        new_u_value = 0
        new_l_value = 0
        # update weights (needs vetorizing)
        for i in range(len(weights)):
            current_weight = weights[i]
            current_zono = pre_zonos[i]
            if current_weight> 0:
                new_u_values += current_weight * current_zono.u_value
                new_l_values += current_weight * current_zono.l_value
            else:
                new_l_values += current_weight * current_zono.u_value
                new_u_values += current_weight * current_zono.l_value
        return Zono(new_u_value, new_u_params, new_l_value, new_l_params)

    def relu_relax(self):
        if self.l_value > 0:
            return self
        elif self.u_value < 0:
            return Zono(0, self.u_params * 0, 0, self.l_params * 0)
        else:
            # use the bound x' > x





    def relax(self, method='relu'):
        # Passing Relu
        zonotopes = []
        lower_bound, upper_bound = self.get_bounds()
        if method == 'relu':
            zonotopes = relu_relaxation(self.a_0, lower_bound, upper_bound)
        else:
            pass
        return zonotopes
    
    def get_bounds(self):
        eps_tensor = torch.Tensor(self.params.shape[1], 1)
        eps_tensor.fill_(self.eps)
        lower_bound, upper_bound = solve_matrix_multiplication_bound(
            self.params, -eps_tensor, eps_tensor)
        return lower_bound, upper_bound