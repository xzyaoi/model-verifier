import torch
from native import Zonotope
from native import SlopeLoss

epochs = 10

def verify(net, inputs, eps, true_label):   
    for i in range(epochs):
        model = Zonotope(net.layers, eps)
        optimizer = torch.optim.SGD(model.slopes, lr=0.2)
        slopeloss = SlopeLoss()
        optimizer.zero_grad()
        u, l = model(inputs)
        loss = slopeloss(u, l, true_label)
        print(loss)
        loss.backward()
        optimizer.step()