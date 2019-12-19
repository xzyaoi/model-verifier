import torch
from native import Zonotope
from native import SlopeLoss

epochs = 200
def verify(net, inputs, eps, true_label):
    for i in range(epochs):
        model = Zonotope(net.layers, eps)
        optimizer = torch.optim.SGD(model.slopes, lr=0.3, momentum=0.95, weight_decay=0.05, nesterov=True)
        slopeloss = SlopeLoss()
        optimizer.zero_grad()
        u, l = model(inputs)
        loss = slopeloss(u, l, true_label)
        print(loss)
        loss.backward()
        upper, lower = u.tolist(), l.tolist()
        upper.remove(upper[true_label])
        result = lower[true_label] > max(upper)
        if result:
            return result
        optimizer.step()
    return False