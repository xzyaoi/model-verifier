import torch
import torch.nn as nn
import torch.nn.functional as F

from new_utils import relu_relax_single_neuro

class Zonotope(object):

    def __init__(self, a_0, eps_params, name="null"):
        self.a_0 = a_0
        self.eps_params = eps_params
        self.name = name
        self.eps = 1

    def get_bound(self):
        self.eps_params = torch.Tensor(self.eps_params)
        positive = F.relu(self.eps_params)
        negative = -F.relu(-self.eps_params)
        upper = positive * self.eps + negative * (-self.eps)
        lower = positive * (-self.eps) + negative * (self.eps)
        return self.a_0 + lower, self.a_0 + upper

    def relax(self, method):
        if method == "relu":
            lower, upper = self.get_bound()
            new_a0, new_eps_params, new_eps = relu_relaxation(
                self.a_0, lower, upper
            )
        else:
            print("method not supported!")

    def linear(self, weight, bias):
        a_0 = F.linear(self.a_0, weight, bias)
        params = F.linear(self.eps_params, weight, bias)
        return Zonotope(a_0, params, "post_linear")


class Layer(object):
    def __init__(self, zonotopes):
        self.zonotopes = zonotopes
    
    def __len__(self):
        return len(self.zonotopes)

    def perform_linear(self, weight, bias):
        a_0 = torch.Tensor([z.a_0 for z in self.zonotopes])
        # shape is dim * k(#eps)
        # for i in self.zonotopes: print(i.eps_params)
        params_map = torch.Tensor([z.eps_params for z in self.zonotopes])
        # params = torch.Tensor([sum(z.eps_params) for z in self.zonotopes])
        new_a_0 = F.linear(a_0, weight, bias)
        new_params = F.linear(params_map.transpose(0,1), weight, bias)
        zonotopes = [Zonotope(a_0, [eps]) for a_0, eps in zip(new_a_0, new_params[0])]
        return Layer(zonotopes)

    def perform_relu(self):
        for index, item in enumerate(self.zonotopes):
            b_0, new_eps_params = relu_relax_single_neuro(item.a_0, item.eps_params)
            self.zonotopes[index] = Zonotope(b_0, new_eps_params)
        return Layer(self.zonotopes)

    def calc_bounds(self):
        lowers = []
        uppers = []
        for each in self.zonotopes:
            lower, upper = each.get_bound()
            lowers.append(lower.data)
            uppers.append(upper.data)
        return lowers, uppers
