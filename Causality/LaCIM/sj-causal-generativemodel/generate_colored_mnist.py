import torch
import os
from PIL import Image
import os
import os.path
import numpy as np
import torch
import random
from PIL import Image
from torchvision import datasets, transforms
import scipy.stats as stats
import copy


def generate_mnist_color(sigma=0.02, env_num=2, color_num=2, env_type=0, test_ratio=0.1):
    train_save_dir = './data/colored_MNIST_%0.2f_env_%d_%d_c_%d_%0.2f/train/' % (
        sigma, env_num, env_type, color_num, test_ratio)
    test_save_dir = './data/colored_MNIST_%0.2f_env_%d_%d_c_%d_%0.2f/test/' % (
        sigma, env_num, env_type, color_num, test_ratio)
    for i in range(10):
        if not os.path.exists('%s%d/' % (train_save_dir, i)):
            os.makedirs('%s%d/' % (train_save_dir, i))
        if not os.path.exists('%s%d/' % (test_save_dir, i)):
            os.makedirs('%s%d/' % (test_save_dir, i))

    a = datasets.MNIST('./data/', train=True, download=True,
                       transform=None)
    a = datasets.MNIST('./data/', train=False, download=True,
                       transform=None)

    train_dir = './data/MNIST/processed/training.pt'
    test_dir = './data/MNIST/processed/test.pt'

    # colors = {
    #     0:(204.0, 0.0, 0.0),
    #     1:(0.0, 153, 153,),
    #     2:(204, 204, 0),
    #     3:(51, 51, 204),
    #     4:(204, 102, 51),
    #     5:(204, 51, 204),
    #     6:(102, 204, 204),
    #     7:(204, 153, 204),
    #     8:(152, 204, 102),
    #     9:(152, 51, 51)
    # }
    if color_num == 2:
        colors = {
            0: (0.0, 204.0, 0.0),
            1: (0.0, 204.0, 0.0),
            2: (0.0, 204.0, 0.0),
            3: (0.0, 204.0, 0.0),
            4: (0.0, 204.0, 0.0),
            5: (204.0, 0.0, 0.0),
            6: (204.0, 0.0, 0.0),
            7: (204.0, 0.0, 0.0),
            8: (204.0, 0.0, 0.0),
            9: (204.0, 0.0, 0.0),
        }
    elif color_num == 5:
        colors = {
            0: (0.0, 204.0, 0.0),
            1: (0.0, 204.0, 0.0),
            2: (204, 204, 0),
            3: (204, 204, 0),
            4: (51, 51, 204),
            5: (51, 51, 204),
            6: (204, 153, 204),
            7: (204, 153, 204),
            8: (204.0, 0.0, 0.0),
            9: (204.0, 0.0, 0.0),
        }
    test_color = (0.5, 0.5, 0.5)
    for key in colors.keys():
        colors[key] = colors[key] / np.array([255.0, 255.0, 255.0])

    if env_num == 1:
        if env_type == 0:
            train_ratio = [0.9]
    elif env_num == 2:
        if env_type == 0:
            train_ratio = [0.9, 0.8]
        elif env_type == 1:
            train_ratio = [0.9, 0.95]
        elif env_type == 3:
            train_ratio = [0.99, 0.95]
        elif env_type == 2:
            train_ratio = [0.9, 0.85]
    elif env_num == 3:
        if env_type == 0:
            train_ratio = [0.9, 0.8, 0.95]
        elif env_type == 1:
            train_ratio = [0.9, 0.8, 0.5]
    elif env_num == 4:
        train_ratio = [0.9, 0.8, 0.7, 0.6]
    elif env_num == 5:
        train_ratio = [0.9, 0.8, 0.7, 0.6, 0.5]

    # training process
    data, target = torch.load(train_dir)
    each_env_num = data.size(0) // env_num
    for i in range(data.size(0)):
        tar = int(target[i])
        img = data[i].numpy()
        img = np.expand_dims(img, 0)
        img = img.repeat(3, 0)

        env_idx = i // each_env_num
        samples = stats.bernoulli.rvs(p=train_ratio[env_idx], size=100)
        if samples[0] == 1:
            mean_color = colors[tar]
        else:
            if color_num == 2:
                mean_color = colors[9 - tar]
            elif color_num == 5:
                if tar <= 1:
                    mean_color = colors[random.choice(list(range(2, 10)))]
                elif 1 < tar <= 3:
                    mean_color = colors[random.choice(
                        [0, 1, 4, 5, 6, 7, 8, 9])]
                elif 3 < tar <= 5:
                    mean_color = colors[random.choice(
                        [0, 1, 2, 3, 6, 7, 8, 9])]
                elif 5 < tar <= 7:
                    mean_color = colors[random.choice(
                        [0, 1, 2, 3, 4, 5, 8, 9])]
                else:
                    mean_color = colors[random.choice(
                        [0, 1, 2, 3, 4, 5, 6, 7])]
        current_color = mean_color + np.random.normal(0.0, sigma, (3,))
        current_color[current_color < 0.0] = 0.0
        current_color[current_color > 1.0] = 1.0
        for s in range(3):
            img[s, :, :] = current_color[s] * img[s, :, :]

        img = Image.fromarray(img.astype('uint8').transpose((1, 2, 0)))
        img.save('%s%d/img_%05d_%0.2f_%d.png' %
                 (train_save_dir, tar, i, train_ratio[env_idx], env_idx))

    # testing process
    data, target = torch.load(test_dir)

    for i in range(data.size(0)):
        tar = int(target[i])
        img = data[i].numpy()
        img = np.expand_dims(img, 0)
        img = img.repeat(3, 0)

        samples = stats.bernoulli.rvs(p=test_ratio, size=100)
        if samples[0] == 1:
            mean_color = colors[tar]
        else:
            if color_num == 2:
                mean_color = colors[9 - tar]
            elif color_num == 5:
                if tar <= 1:
                    mean_color = colors[random.choice(list(range(2, 10)))]
                elif 1 < tar <= 3:
                    mean_color = colors[random.choice(
                        [0, 1, 4, 5, 6, 7, 8, 9])]
                elif 3 < tar <= 5:
                    mean_color = colors[random.choice(
                        [0, 1, 2, 3, 6, 7, 8, 9])]
                elif 5 < tar <= 7:
                    mean_color = colors[random.choice(
                        [0, 1, 2, 3, 4, 5, 8, 9])]
                else:
                    mean_color = colors[random.choice(
                        [0, 1, 2, 3, 4, 5, 6, 7])]
        #mean_color = colors[random.choice(list(range(10)))]
        current_color = mean_color + np.random.normal(0.0, sigma, (3,))
        current_color[current_color < 0.0] = 0.0
        current_color[current_color > 1.0] = 1.0
        for s in range(3):
            img[s, :, :] = current_color[s] * img[s, :, :]

        img = Image.fromarray(img.astype('uint8').transpose((1, 2, 0)))
        img.save('%s%d/img_%05d_%0.2f_%d.png' %
                 (test_save_dir, tar, i, test_ratio, 0))


def generate_mnist_color_IRM(sigma=0.0, env_num=2, color_num=2, env_type=0, test_ratio=0.1):
    train_save_dir = './data/colored_MNIST_IRM_%0.2f_env_%d_%d_c_%d_%0.2f/train/' % (
        sigma, env_num, env_type, color_num, test_ratio)
    test_save_dir = './data/colored_MNIST_IRM_%0.2f_env_%d_%d_c_%d_%0.2f/test/' % (
        sigma, env_num, env_type, color_num, test_ratio)
    for i in range(10):
        if not os.path.exists('%s%d/' % (train_save_dir, i)):
            os.makedirs('%s%d/' % (train_save_dir, i))
        if not os.path.exists('%s%d/' % (test_save_dir, i)):
            os.makedirs('%s%d/' % (test_save_dir, i))

    a = datasets.MNIST('./data/', train=True, download=True,
                       transform=None)
    a = datasets.MNIST('./data/', train=False, download=True,
                       transform=None)

    train_dir = './data/MNIST/processed/training.pt'
    test_dir = './data/MNIST/processed/test.pt'

    # colors = {
    #     0:(204.0, 0.0, 0.0),
    #     1:(0.0, 153, 153,),
    #     2:(204, 204, 0),
    #     3:(51, 51, 204),
    #     4:(204, 102, 51),
    #     5:(204, 51, 204),
    #     6:(102, 204, 204),
    #     7:(204, 153, 204),
    #     8:(152, 204, 102),
    #     9:(152, 51, 51)
    # }
    if color_num == 2:
        colors = {
            0: (0.0, 255.0, 0.0),
            1: (0.0, 255.0, 0.0),
            2: (0.0, 255.0, 0.0),
            3: (0.0, 255.0, 0.0),
            4: (0.0, 255.0, 0.0),
            5: (255.0, 0.0, 0.0),
            6: (255.0, 0.0, 0.0),
            7: (255.0, 0.0, 0.0),
            8: (255.0, 0.0, 0.0),
            9: (255.0, 0.0, 0.0),
        }
    green = [0.0, 1.0, 0.0]
    red = [1.0, 0.0, 0.0]
    test_color = (0.5, 0.5, 0.5)
    for key in colors.keys():
        colors[key] = colors[key] / np.array([255.0, 255.0, 255.0])

    if env_num == 1:
        if env_type == 0:
            train_ratio = [0.9]  # sijie 代码dataloader中的u
    elif env_num == 2:
        if env_type == 0:
            train_ratio = [0.9, 0.8]
        elif env_type == 1:
            train_ratio = [0.9, 0.95]
        elif env_type == 2:
            train_ratio = [0.9, 0.85]
    elif env_num == 3:
        if env_type == 0:
            train_ratio = [0.9, 0.8, 0.7]
    elif env_num == 4:
        if env_type == 0:
            train_ratio = [0.9, 0.8, 0.7, 0.6]
        elif env_type == 1:
            train_ratio = [0.9, 0.85, 0.8, 0.75]

    # training process
    data, target = torch.load(train_dir)
    real_target = copy.deepcopy(target)
    each_env_num = data.size(0) // env_num
    green_counter = 0
    total_counter = 0
    for i in range(data.size(0)):
        real_tar = int(target[i])
        samples = stats.bernoulli.rvs(p=0.25, size=50)
        if samples[0] == 1:
            target[i] = 9 - real_tar
        else:
            target[i] = real_tar

    for i in range(data.size(0)):
        tar = int(target[i])
        img = data[i].numpy()
        img = np.expand_dims(img, 0)
        img = img.repeat(3, 0)

        env_idx = i // each_env_num
        samples = stats.bernoulli.rvs(p=[env_idx], size=100)
        if samples[0] == 1:
            if tar > 4:
                mean_color = red
            else:
                mean_color = green
        else:
            if tar > 4:
                mean_color = green
            else:
                mean_color = red
        # print(mean_color, mean_color[1])
        if real_target[i] > 4 and env_idx == 0:
            total_counter = total_counter + 1

        if mean_color[1] == 1.0 and env_idx == 0 and real_target[i] > 4:
            green_counter = green_counter + 1
        current_color = mean_color  # + np.random.normal(0.0, sigma, (3,))
        # current_color[current_color < 0.0] = 0.0
        # current_color[current_color > 1.0] = 1.0
        for s in range(3):
            img[s, :, :] = current_color[s] * img[s, :, :]

        img = Image.fromarray(img.astype('uint8').transpose((1, 2, 0)))
        img.save('%s%d/img_%05d_%d_%0.2f_%d.png' % (train_save_dir, tar,
                 i, int(real_target[i]), train_ratio[env_idx], env_idx))

    print('green counter for train', green_counter, total_counter)

    # testing process
    data, target = torch.load(test_dir)
    real_target = copy.deepcopy(target)
    for i in range(data.size(0)):
        real_tar = int(target[i])
        samples = stats.bernoulli.rvs(p=0.25, size=50)
        if samples[0] == 1:
            target[i] = 9 - real_tar
        else:
            target[i] = real_tar

    green_counter = 0
    for i in range(data.size(0)):
        tar = int(target[i])

        img = data[i].numpy()
        img = np.expand_dims(img, 0)
        img = img.repeat(3, 0)

        samples = stats.bernoulli.rvs(p=test_ratio, size=100)
        if samples[0] == 1:
            if tar > 4:
                mean_color = red
            else:
                mean_color = green
        else:
            if tar > 4:
                mean_color = green
            else:
                mean_color = red

        if mean_color[1] == 1.0:
            green_counter = green_counter + 1
        # mean_color = colors[random.choice(list(range(10)))]
        current_color = mean_color  # + np.random.normal(0.0, sigma, (3,))
        #
        # current_color[current_color < 0.0] = 0.0
        # current_color[current_color > 1.0] = 1.0
        for s in range(3):
            img[s, :, :] = current_color[s] * img[s, :, :]

        img = Image.fromarray(img.astype('uint8').transpose((1, 2, 0)))
        img.save('%s%d/img_%05d_%d_%0.2f_0.png' %
                 (test_save_dir, tar, i, int(real_target[i]), test_ratio))
    print('green counter for test', green_counter)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='dataset')
    parser.add_argument('--sigma', type=float, default=0.02)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--percent', type=int, default=10)
    parser.add_argument('--env_num', type=int, default=2)
    parser.add_argument('--env_type', type=int, default=0)
    parser.add_argument('--color_num', type=int, default=2)

    args = parser.parse_args()
    # generate_mnist_color_IRM(sigma=args.sigma, env_num=args.env_num, env_type=args.env_type, color_num=args.color_num,
    #                      test_ratio=args.test_ratio)
    generate_mnist_color(sigma=args.sigma, env_num=args.env_num, env_type=args.env_type, color_num=args.color_num,
                         test_ratio=args.test_ratio)

    # generate_mnist_color_f(args.sigma)
    # generate_mnist_color_s(args.sigma, args.percent)
    # generate_mnist_color_test(args.sigma)

    # generate_mnist_color_background(args.sigma, fixed_color=1)
    # generate_mnist_color_background(args.sigma)
    # generate_mnist_color_background(args.sigma, args.percent)
    # generate_mnist_color_f(args.sigma) # full shuffle
    # generate_mnist_color_s(args.sigma, args.percent) # training part shuffle 10%
    # generate_mnist_color(args.sigma)
    # generate_mnist_color_test(args.sigma)
    # generate_mnist_test()
