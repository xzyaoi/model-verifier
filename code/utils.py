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


def merge_neuros(neuros):
    a_0 = torch.Tensor([x.a_0 for x in neuros]).reshape(1, len(neuros))
    params = torch.Tensor([x.params for x in neuros]).reshape(1, len(neuros))
    return Zonotope(a_0, params, 1)


def relu_relaxation(input_a0, lower_bound, upper_bound, xs):
    def calc_each_form(a_0, l, u,x):
        if (u <= 0):
            return 0, 0, 0
        elif (l >= 0):
            return x, 0, 0
        else:
            slope = u/(u-l)
            # print("lower: "+str(l.data) +" upper: "+str(u.data) + "====>" +"slope:" + str(slope.data))
            # print("lower: y="+str(slope.data)+"x")
            # print("upper: y="+str(slope.data)+"x-"+str(slope.data*l.data))
            return slope * a_0 - slope * l / 2, -slope * l / 2, 1
    _, bound_length = lower_bound.shape
    # we will have #bound_length zonotopes
    zonotopes = []
    for i in range(bound_length):
        new_a0, new_params, new_eps = calc_each_form(
            input_a0[0, i], lower_bound[0, i], upper_bound[0, i], xs[0,i])
        zonotopes.append(Zonotope(new_a0, new_params, new_eps))
    return zonotopes


class Zonotope(object):
    def __init__(self, a_0, params, eps, name="null"):
        self.a_0 = a_0
        self.params = params
        self.eps = eps
        self.name = name

    def print(self):
        print("---"+self.name +"---")
        print("---a0 shape: "+str(self.a_0.shape)+"---")
        print("---params shape: "+str(self.params.shape)+"---")
        print("---eps:"+str(self.eps)+"---")
    
    def relax(self,xs, method='relu'):
        # Passing Relu
        zonotopes = []
        lower_bound, upper_bound = self.get_bounds()
        if method == 'relu':
            zonotopes = relu_relaxation(self.a_0, lower_bound, upper_bound, xs)
        else:
            pass
        return merge_neuros(zonotopes)

    def plugin(self, xs):
        def calc_y_prime(u,l,a0,x):
            if u<0:
                return 0
            if l>0:
                return x
            else:
                return u/(u-l)
        lower_bound, upper_bound = self.get_bounds()
        if method == 'relu':
            zonotopes = relu_relaxation(self.a_0, lower_bound, upper_bound)
        else:
            pass
        _, bound_length = lower_bound.shape
        for i in range(bound_length):
            pass

    def linear(self, weight, bias):
        a_0 = F.linear(self.a_0, weight, bias)
        params = torch.matmul(self.params, weight.transpose(0, 1))
        return Zonotope(a_0, params, 1)

    def get_bounds(self):
        eps_tensor = torch.Tensor(self.params.shape[1], 1)
        eps_tensor.fill_(self.eps)
        lower_bound, upper_bound = solve_matrix_multiplication_bound(
            self.params, -eps_tensor, eps_tensor)
        return self.a_0 + lower_bound, self.a_0 + upper_bound

    def get_final_bounds(self):
        positive = F.relu(self.params)
        negative = -F.relu(-self.params)
        upper = positive * self.eps + negative * (-self.eps)
        lower = positive * (-self.eps) + negative * (self.eps)
        return self.a_0 + lower, self.a_0 + upper
