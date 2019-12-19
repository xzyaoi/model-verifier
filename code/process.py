import torch
from native import Zonotope
from native import SlopeLoss

epochs = 200

def verify(net, inputs, eps, true_label):
    with torch.autograd.set_detect_anomaly(True):
        model = Zonotope(net.layers, eps)
        slopeloss = SlopeLoss()
        with torch.no_grad():
            # init slopes with u/(u-l)
            u,l = model(inputs)
        param_groups = []
        current_lr = 0.02
        for each in model.slopes:
            current_lr = current_lr * 4
            param_groups.append({
                'params': each,
                'lr': current_lr
            })
        optimizer = torch.optim.Adam(param_groups, lr=0.001)
        # optimizer = torch.optim.SGD(param_groups, momentum=0.95, weight_decay=0.05, nesterov=True)
        for i in range(epochs):
            optimizer.zero_grad()
            u, l = model(inputs.detach())
            # print(l)
            # print(u)
            loss, number_of_violations = slopeloss(u, l, true_label)
            loss.backward()
            # scheduler.step(number_of_violations)
            upper, lower = u.tolist(), l.tolist()
            upper.remove(upper[true_label])
            result = lower[true_label] > max(upper)
            if result:
                print(result)
                return result
                break
            optimizer.step()
            # restrict slopes to be between 0 and 1
            with torch.no_grad():
                for param in model.slopes:
                    param.clamp_(0, 1)
            # print(model.slopes)
        return False