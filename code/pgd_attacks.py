import torch
import torch.nn as nn
from networks import FullyConnected, Conv
import numpy as np
DEVICE='cpu'
INPUT_SIZE=28

def fgsm_(model, x, target, eps, targeted=True, device='cpu', clip_min=None, clip_max=None):
    """Internal process for all FGSM and PGD attacks."""
    # create a copy of the input, remove all previous associations to the compute graph...
    input_ = x.clone().detach_()
    # ... and make sure we are differentiating toward that variable
    input_.requires_grad_()

    # run the model and obtain the loss
    logits = model(input_)
    target = torch.LongTensor([target]).to(device)
    model.zero_grad()
    loss = nn.CrossEntropyLoss()(logits, target)
    loss.backward()

    # perfrom either targeted or untargeted attack
    if targeted:
        out = input_ - eps * input_.grad.sign()
    else:
        out = input_ + eps * input_.grad.sign()

    # if desired clip the ouput back to the image domain
    if (clip_min is not None) or (clip_max is not None):
        out.clamp_(min=clip_min, max=clip_max)
    return out


def pgd_(model, x, target, k, eps, eps_step, targeted=True, device='cpu', clip_min=None, clip_max=None):
    x_min = x - eps
    x_max = x + eps

    # generate a random point in the +-eps box around x
    x = torch.rand_like(x)
    x = (x*2*eps - eps)

    for i in range(k):
        # FGSM step
        x = fgsm_(model, x, target, eps_step, targeted, device)
        # Projection Step
        x = torch.max(x_min, x)
        x = torch.min(x_max, x)
    # if desired clip the ouput back to the image domain
    if (clip_min is not None) or (clip_max is not None):
        x.clamp_(min=clip_min, max=clip_max)
    pred = torch.argmax(net(x))
    if label != pred:
        print(label, pred)
        print('False')
    return x


def pgd_untargeted(model, x, label, k, eps, eps_step, device='cpu', clip_min=None, clip_max=None, **kwargs):
    return pgd_(model, x, label, k, eps, eps_step, targeted=False, device=device, clip_min=clip_min, clip_max=clip_max, **kwargs)


def pgd_untargeted_batched(model, x_batch, y_batch, k, eps, eps_step, device='cpu', clip_min=None, clip_max=None, **kwargs):
    n = x_batch.size()[0]
    xprime_batch_list = []
    for i in range(n):
        x = x_batch[i, ...]
        y = y_batch[i]
        xprime = pgd_untargeted(model, x, y, k, eps,
                                eps_step, device, clip_min, clip_max, **kwargs)
        xprime_batch_list.append(xprime)
    xprime_batch_tensor = torch.stack(xprime_batch_list)
    assert x_batch.size() == xprime_batch_tensor.size()
    return xprime_batch_tensor


if __name__ == "__main__":
    k = 10000000
    netname = 'fc1'
    spec = "../test_cases/"+netname+"/img0_0.06000.txt"

    if netname == 'fc1':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 10]).to(DEVICE)
    elif netname == 'fc2':
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 50, 10]).to(DEVICE)
    elif netname == 'fc3':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif netname == 'fc4':
        net = FullyConnected(DEVICE, INPUT_SIZE, [
                             100, 100, 100, 10]).to(DEVICE)
    elif netname == 'fc5':
        net = FullyConnected(DEVICE, INPUT_SIZE, [
                             400, 200, 100, 100, 10]).to(DEVICE)
    elif netname == 'conv1':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1)], [
                   100, 10], 10).to(DEVICE)
    elif netname == 'conv2':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)], [
                   100, 10], 10).to(DEVICE)
    elif netname == 'conv3':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [
                   150, 10], 10).to(DEVICE)
    elif netname == 'conv4':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)], [
                   100, 100, 10], 10).to(DEVICE)
    elif netname == 'conv5':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [
                   100, 100, 10], 10).to(DEVICE)
    net.load_state_dict(torch.load('../mnist_nets/%s.pt' %
                                   netname, map_location=torch.device(DEVICE)))

    with open(spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(spec[:-4].split('/')[-1].split('_')[-1])

    inputs = torch.FloatTensor(pixel_values).view(
        1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    y = pgd_untargeted(net, inputs, true_label, k, eps, eps_step=0.00001)

