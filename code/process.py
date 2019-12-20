import torch
from module import Zonotope
from module import SlopeLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

# epochs = 20000
epochs = 20000

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def _get_verify_result(u,l,true_label):
    # print(u)
    # print(l)
    # print([i for i in range(10) if u[i] > l and i != true_label])
    u.remove(u[true_label])
    return l > max(u)

def verify(net, inputs, eps, true_label):
    with torch.autograd.set_detect_anomaly(True):
        model = Zonotope(net.layers, eps, true_label)
        slopeloss = SlopeLoss()
        with torch.no_grad():
            # init slopes with u/(u-l)
            u,l = model(inputs)
            result = _get_verify_result(model.uppers.copy(),model.true_lower,true_label)
            if result:
                return True
        param_groups = []
        current_lr = 0.22
        for each in model.slopes:
            current_lr = current_lr * 1.15
            param_groups.append({
                'params': each,
                'lr': current_lr
            })
        optimizer = torch.optim.Adam(param_groups)
        # optimizer = torch.optim.SGD(param_groups, momentum=0.95, weight_decay=0.05, nesterov=True)
        scheduler = ReduceLROnPlateau(optimizer, 'max',verbose=False,patience=10, factor=0.95)
        for i in range(epochs):
            optimizer.zero_grad()
            u,l = model(inputs)
            result = _get_verify_result(model.uppers.copy(),model.true_lower,true_label)
            if result:
                return True
            loss = slopeloss(u, l, true_label)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for param in model.slopes:
                    param.clamp_(0, 1)
        return False