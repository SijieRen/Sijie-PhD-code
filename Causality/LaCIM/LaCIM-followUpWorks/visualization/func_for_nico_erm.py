import torch
import random
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
from scipy.ndimage.filters import gaussian_filter1d

import torch
import torchvision
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms


SQUEEZENET_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
SQUEEZENET_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def preprocess(img, size=224):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=SQUEEZENET_MEAN.tolist(),
                             std=SQUEEZENET_STD.tolist()),
        transforms.Lambda(lambda x: x[None]),
        # Add a batch dimension in the first position of the tensor:
        # aka, a tensor of shape (H, W, C) will become -> (1, H, W, C).
    ])
    # return transforms(img).unsqueeze(0)  也可以使用这种方式增加第一维
    return transform(img)


def deprocess(img, should_rescale=True):
    """ 
    De-processes a Pytorch tensor from the output of the CNN model 
    to become a PIL JPG Image 
    """
    transform = transforms.Compose([
        # Remove the batch dimension at the first position. A tensor of dims (1, H, W, C) will become -> (H, W, C).
        transforms.Lambda(lambda x: x[0]),
        # Normalize the standard deviation
        transforms.Normalize(mean=[0, 0, 0], std=(
            1.0 / SQUEEZENET_STD).tolist()),
        transforms.Normalize(mean=(-SQUEEZENET_MEAN).tolist(),
                             std=[1, 1, 1]),  # Normalize the mean
        # Rescale all the values in the tensor so that they lie in the interval [0, 1] to prepare for transforming it into image pixel values.
        transforms.Lambda(
            rescale) if should_rescale else transforms.Lambda(lambda x: x),
        transforms.ToPILImage(),
    ])
    return transform(img)


def rescale(x):
    """ A function used internally inside `deprocess`.
        Rescale elements of x linearly to be in the interval [0, 1]
        with the minimum element(s) mapped to 0, and the maximum element(s)
        mapped to 1.
    """
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled

# Saliency Maps 函数


def compute_saliency_maps_erm(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.
    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.
    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    model.eval()
    X.requires_grad_()
    # scores = model(X, is_train=2, is_debug=1)
    scores = model(X)

    # loss = nn.CrossEntropyLoss(scores, y)  # 这么写是错的，因为nn.CrossEntropyLoss是一个类，要先初始化
    loss = nn.functional.cross_entropy(scores, y)
    loss.backward()
    # X.grad.shape: torch.Size([5, 3, 224, 224])
    # print("saliency grad", X.grad)
    saliency = torch.max(torch.abs(X.grad), 1)[0]  # 第一个元素是想要的最大值，第二个元素是index
    # saliency.shape: torch.Size([5, 224, 224])

    return saliency
# 显示 Saliency Maps


def show_saliency_maps(X, y, model):
    # Convert X and y from numpy arrays to Torch Tensors
    X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
    # X_tensor.shape: torch.Size([5, 3, 224, 224])
    y_tensor = torch.LongTensor(y)

    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(X_tensor, y_tensor, model)

    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.numpy()
    N = X.shape[0]
    for i in range(N):
        plt.subplot(2, N, i + 1)
        plt.imshow(X[i])
        plt.axis('off')
        plt.title(class_names[y[i]])
        plt.subplot(2, N, N + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.bone)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
    plt.show()
#
# hot 从黑平滑过度到红、橙色和黄色的背景色，然后到白色。
# cool 包含青绿色和品红色的阴影色。从青绿色平滑变化到品红色。
# gray 返回线性灰度色图。
# bone 具有较高的蓝色成分的灰度色图。该色图用于对灰度图添加电子的视图。
# white 全白的单色色图。
# spring 包含品红和黄的阴影颜色。
# summer 包含绿和黄的阴影颜色。
# autumn 从红色平滑变化到橙色，然后到黄色。
# winter 包含蓝和绿的阴影色。
#

# show_saliency_maps(X, y)


# Fooling Images
def make_fooling_image_erm(X, target_y, model):
    """
    Generate a fooling image that is close to X, but that the model classifies
    as target_y.
    Inputs:
    - X: Input image; Tensor of shape (1, 3, 224, 224)
    - target_y: An integer in the range [0, 1000)
    - model: A pretrained CNN
    Returns:
    - X_fooling: An image that is close to X, but that is classifed as target_y
    by the model.
    """
    # Initialize our fooling image to the input image, and make it require gradient
    X_fooling = X.clone()
    X_fooling = X_fooling.requires_grad_()

    learning_rate = 1

    # When computing an update step, first normalize the gradient: dX = learning_rate * g / ||g||_2
    for i in range(100):
        # shape: torch.Size([1, 1000])
        # scores = model(X_fooling, is_train=2, is_debug=1)
        scores = model(X_fooling)
        # print("scores: ", scores)
        index = torch.argmax(scores, 1)
        _, index = scores.data.max(dim=1)
        # print("index", index)
        print("i: ", i)
        if target_y == index:
            break
        target_score = scores[0, target_y]
        # print("target_score", target_score)
        target_score.backward()
        grad = X.grad
        print("x_fooling 是否是叶张量", X.is_leaf)
        # if grad:
            # print("grad", grad)
        X_fooling.data += learning_rate * (grad / grad.norm())
        # else:
        #     break
        X.grad.zero_()

    # X_fooling_np = deprocess(X_fooling.clone())
    # X_fooling_np = np.asarray(X_fooling_np).astype(np.uint8)

    return X_fooling


# 产生图片

def show_fooling_maps(X, target_y, model):
    idx = 2
    target_y = 1

    X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
    X_fooling = make_fooling_image(X_tensor[idx:idx+1], target_y, model)

    scores = model(X_fooling)
    assert target_y == scores.data.max(
        1)[1][0].item(), 'The model is not fooled!'

    # 显示
    X_fooling_np = deprocess(X_fooling.clone())
    X_fooling_np = np.asarray(X_fooling_np).astype(np.uint8)

    plt.subplot(1, 4, 1)
    plt.imshow(X[idx])
    plt.title(class_names[y[idx]])
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(X_fooling_np)
    plt.title(class_names[target_y])
    plt.axis('off')

    plt.subplot(1, 4, 3)
    X_pre = preprocess(Image.fromarray(X[idx]))
    diff = np.asarray(deprocess(X_fooling - X_pre, should_rescale=False))
    plt.imshow(diff)
    plt.title('Difference')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    diff = np.asarray(
        deprocess(10 * (X_fooling - X_pre), should_rescale=False))
    plt.imshow(diff)
    plt.title('Magnified difference (10x)')
    plt.axis('off')

    plt.gcf().set_size_inches(12, 5)
    plt.show()
