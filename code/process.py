import torch
from native import Zonotope
from native import SlopeLoss

epochs = 100

def verify(net, inputs, eps, true_label):
    with torch.autograd.set_detect_anomaly(True):
        model = Zonotope(net.layers, eps)
        slopeloss = SlopeLoss()
        with torch.no_grad():
            # init slopes with u/(u-l)
            u,l = model(inputs)
        param_groups = []
        for each in model.slopes:
            param_groups.append({
                'params': each,
            })
        # optimizer = torch.optim.Adam(param_groups,lr=0.3)
        optimizer = torch.optim.SGD(param_groups,lr=0.03)
        for i in range(epochs):
            optimizer.zero_grad()
            u, l = model(inputs)
            loss, number_of_violations = slopeloss(u, l, true_label)
            loss.backward()
            print(u)
            print(l)
            print(loss)
            print("number_of_violations %d" % number_of_violations)
            upper, lower = u.tolist(), l.tolist()
            upper.remove(upper[true_label])
            result = lower[true_label] > max(upper)
            if result:
                print(result)
                return result
                break
            optimizer.step()
        return False