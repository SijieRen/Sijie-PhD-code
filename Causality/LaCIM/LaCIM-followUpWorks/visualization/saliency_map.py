import torch
from torch.autograd import Variable

import pdb

"""
Use image(x) and label(y) to compute correct saliency map
Args:
    x - input images Varialbe: (B, C, H, W) = (B, 3, H, W)
    y - labels, should be a LongTensor: (N,), N is the number of input images
    model - a pretrained model
Return:
    saliency maps, a tensor of shape (B, H, W), one per input image
"""
def compute_saliency_maps(x, y, model):
    if not isinstance(x, Variable):
        x = Variable(x, requires_grad = True)
    elif not x.requires_grad:
        x.requires_grad = True

    if not isinstance(y, Variable):
        y = Variable(y)

    # forward pass
    scores = model(x)

    scores = scores.gather(1, y.view(-1, 1)).squeeze()

    # backward pass
    scores.backward(torch.ones(scores.shape))

    saliency_maps = x.grad.data

    saliency_maps = saliency_maps.abs()
    saliency_maps, idx = torch.max(saliency_maps, dim = 1)    # get max abs from all (3) channels

    return saliency_maps

"""
Use cross entropy to compute saliency maps
"""
def compute_saliency_maps_crossentropy(x, y, model):
    if not isinstance(x, Variable):
        x = Variable(x, requires_grad = True)
    elif not x.requires_grad:
        x.requires_grad = True

    if not isinstance(y, Variable):
        y = Variable(y) # shape: (5)

    # forward pass
    scores = model(x)   # (5 * 1000)

    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(scores, y)
    loss.backward()

    saliency_maps = x.grad.data
    saliency_maps = saliency_maps.abs()

    saliency_maps, idx = torch.max(saliency_maps, dim = 1)  # get max abs from all (3) channels

    return saliency_maps

"""
Visualization part
"""
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import pdb

def postprocess(img):
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img.astype(np.uint8)

    return img

"""
generate and display saliency maps
images - image tensors (B, C, H, W)
saliency_maps - the same shape as images
"""
def show_saliency_maps(images, saliency_maps):
    N = images.shape[0] # number of images

    for i in range(N):
        plt.subplot(2, N, i + 1)
        plt.imshow(postprocess(images[i].data.numpy().transpose(1, 2, 0)))
        plt.axis('off')
        plt.subplot(2, N, N + i + 1)
        plt.imshow(postprocess(saliency_maps[i].numpy()), cmap = plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
    plt.show()


def compute_saliency_maps_lacim(x, y, model):
    if not isinstance(x, Variable):
        x = Variable(x, requires_grad = True)
    elif not x.requires_grad:
        x.requires_grad = True

    if not isinstance(y, Variable):
        y = Variable(y)

    # forward pass
    # scores = model(x)
    # for LaCIM
    scores = model(x, is_train=2, is_debug=0)
    

    scores = scores.gather(1, y.view(-1, 1)).squeeze()
    # scores = scores.squeeze()

    # backward pass
    scores.backward(torch.ones(scores.shape))

    saliency_maps = x.grad.data

    saliency_maps = saliency_maps.abs()
    saliency_maps, idx = torch.max(saliency_maps, dim = 1)    # get max abs from all (3) channels

    return saliency_maps

def compute_saliency_maps_erm(x, y, model):
    if not isinstance(x, Variable):
        x = Variable(x, requires_grad = True)
    elif not x.requires_grad:
        x.requires_grad = True

    if not isinstance(y, Variable):
        y = Variable(y)

    # forward pass
    # for erm
    scores = model(x)
    # for LaCIM
    # _, scores = model(x, is_train=1, is_debug=1)

    # scores = scores.squeeze()
    scores = scores.gather(1, y.view(-1, 1)).squeeze()

    # backward pass
    scores.backward(torch.ones(scores.shape))

    saliency_maps = x.grad.data

    saliency_maps = saliency_maps.abs()
    saliency_maps, idx = torch.max(saliency_maps, dim = 1)    # get max abs from all (3) channels

    return saliency_maps