import torch
from module import Zonotope
from module import SlopeLoss

epochs = 1

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def _get_verify_result(u,l,true_label):
    print(u)
    print(l)
    upper, lower = u.tolist(), l.tolist()
    upper.remove(upper[true_label])
    return lower[true_label] > max(upper)

def verify(net, inputs, eps, true_label):
    with torch.autograd.set_detect_anomaly(True):
        model = Zonotope(net.layers, eps)
        slopeloss = SlopeLoss()
        with torch.no_grad():
            # init slopes with u/(u-l)
            u,l = model(inputs)
            result = _get_verify_result(u,l,true_label)
            if result:
                return result
        param_groups = []
        current_lr = 0.25
        for each in model.slopes:
            current_lr = current_lr * 1
            param_groups.append({
                'params': each,
                'lr': current_lr
            })
        optimizer = torch.optim.Adam(param_groups)
        # optimizer = torch.optim.SGD(param_groups, momentum=0.95, weight_decay=0.05, nesterov=True)
        for i in range(epochs):
            optimizer.zero_grad()
            u, l = model(inputs.detach())
            result = _get_verify_result(u,l,true_label)
            if result:
                return result
            loss = slopeloss(u, l, true_label)
            loss.backward()
            optimizer.step()
            # restrict slopes to be between 0 and 1
            with torch.no_grad():
                for param in model.slopes:
                    param.clamp_(0, 1)
        return False