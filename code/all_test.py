from zonotope import Layer, Zonotope
import torch

def test_linear():
    z1 = Zonotope(1, [0.6])
    z2 = Zonotope(-1, [0.5])
    layer = Layer([z1,z2])
    weight = torch.Tensor([[1,-1], [-1,1]])
    bias = torch.Tensor([0.01,0.001])
    output = layer.perform_linear(weight=weight, bias=bias)
    print([z.a_0 for z in output.zonotopes])
    print([z.eps_params for z in output.zonotopes])
    output = output.perform_relu()
    print([z.a_0 for z in output.zonotopes])
    print([z.eps_params for z in output.zonotopes])
    # todo: check the final linear
    new_weight = torch.Tensor(3,2)
    new_weight.fill_(1)
    output = output.perform_linear(weight=new_weight, bias=None, after_relu=True)
    print([z.a_0 for z in output.zonotopes])
    print([z.eps_params for z in output.zonotopes])
    


if __name__ == "__main__":
    test_linear()