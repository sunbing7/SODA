import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

from models.selector import *


def split_model(ori_model, model_name, split_layer=6):
    '''
    split given model from the dense layer before logits
    Args:
        ori_model:
        model_name: model name
    Returns:
        splitted models: 2-5
    '''
    if model_name == 'resnet18' or model_name == 'resnet50':
        if split_layer == 6:    #last
            modules = list(ori_model.children())
            module1 = modules[:2]
            module2 = modules[2:6]
            module3 = [modules[6]]

            model_1st = nn.Sequential(*[*module1, Relu(), *module2, Avgpool2d(), Flatten()])
            model_2nd = nn.Sequential(*module3)

        elif split_layer == 1:  #shallow
            modules = list(ori_model.children())
            module1 = modules[:2]
            module2 = modules[2:6]
            module3 = [modules[6]]

            model_1st = nn.Sequential(*[*module1, Relu()])
            model_2nd = nn.Sequential(*[*module2, Avgpool2d(), Flatten(), *module3])

        elif split_layer == 3:    #mid
            modules = list(ori_model.children())
            module1 = modules[:2]
            module2 = modules[2:4]
            module3 = modules[4:6]
            module4 = [modules[6]]

            model_1st = nn.Sequential(*[*module1, Relu(), *module2])
            model_2nd = nn.Sequential(*[*module3, Avgpool2d(), Flatten(), *module4])

        elif split_layer == 5:    #second last
            modules = list(ori_model.children())
            module1 = modules[:2]
            module2 = modules[2:5]
            module3 = modules[5:6]
            module4 = [modules[6]]

            model_1st = nn.Sequential(*[*module1, Relu(), *module2, *module3])
            model_2nd = nn.Sequential(*[Avgpool2d(), Flatten(), *module4])

        elif split_layer == 9:
            modules = list(ori_model.children())
            module1 = modules[0:9]
            module2 = [modules[-1]]

            model_1st = nn.Sequential(*module1)
            model_2nd = nn.Sequential(*module2)

    elif model_name == 'MobileNet':
        if split_layer == 3:
            modules = list(ori_model.children())
            module1 = modules[:2]
            module2 = [modules[2]]
            module3 = [modules[3]]

            model_1st = nn.Sequential(*[*module1, Relu(), *module2, Avgpool2d_n(poolsize=7), Flatten()])
            model_2nd = nn.Sequential(*module3)
        if split_layer == 6:
            modules = list(ori_model.children())
            module1 = modules[0:6]
            module2 = [modules[-1]]

            model_1st = nn.Sequential(*module1)
            model_2nd = nn.Sequential(*module2)

    elif model_name == 'MobileNetV2':
        if split_layer == 4:
            modules = list(ori_model.children())
            module1 = modules[:2]
            module2 = [modules[2]]
            module3 = modules[3:5]
            module4 = [modules[5]]

            model_1st = nn.Sequential(*[*module1, Relu(), *module2, *module3, Relu(), Avgpool2d(), Flatten()])
            model_2nd = nn.Sequential(*module4)
        if split_layer == 9:
            modules = list(ori_model.children())
            module1 = modules[0:9]
            module2 = [modules[-1]]

            model_1st = nn.Sequential(*module1)
            model_2nd = nn.Sequential(*module2)
    elif model_name == 'vgg11_bn':
        if split_layer == 1:
            modules = list(ori_model.children())

            module1 = [modules[0]]
            module2 = [modules[1]]

            model_1st = nn.Sequential(*[*module1, Flatten()])
            model_2nd = nn.Sequential(*module2)
        if split_layer == 2:
            modules = list(ori_model.children())
            module1 = modules[:2]
            module2 = [modules[-1]]

            model_1st = nn.Sequential(*module1)
            model_2nd = nn.Sequential(*module2)
    elif model_name == 'alexnet':
        if split_layer == 6:
            modules = list(ori_model.children())
            module1 = modules[:5]
            module2 = modules[5:7]
            module3 = [modules[-1]]

            model_1st = nn.Sequential(*[*module1, Flatten(), *module2])
            model_2nd = nn.Sequential(*module3)
    elif model_name == 'shufflenetv2':
        if split_layer == 6:
            modules = list(ori_model.children())
            sub_modules = list(modules[-1])
            module0 = [modules[0]]
            module1 = modules[1:6]
            module2 = [sub_modules[0]]
            module3 = [sub_modules[1]]

            model_1st = nn.Sequential(*[*module0, *module1, Avgpool2d_n(poolsize=7), Flatten(), *module2])
            model_2nd = nn.Sequential(*module3)
    elif model_name == 'densenet':
        if split_layer == 9:
            modules = list(ori_model.children())
            module1 = modules[:9]
            module2 = [modules[-1]]

            model_1st = nn.Sequential(*[*module1, Relu(), Avgpool2d_n(poolsize=4), Flatten()])
            model_2nd = nn.Sequential(*module2)
    else:
        return None, None

    return model_1st, model_2nd


class Relu(nn.Module):
    def __init__(self):
        super(Relu, self).__init__()

    def forward(self, x):
        x = F.relu(x)
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class Avgpool2d(nn.Module):
    def __init__(self):
        super(Avgpool2d, self).__init__()

    def forward(self, x):
        x = F.avg_pool2d(x, 4)
        return x


class Avgpool2d_2(nn.Module):
    def __init__(self):
        super(Avgpool2d_2, self).__init__()

    def forward(self, x):
        x = F.avg_pool2d(x, 2)
        return x

class Avgpool2d_n(nn.Module):
    def __init__(self, poolsize=2):
        super(Avgpool2d_n, self).__init__()
        self.poolsize = poolsize
    def forward(self, x):
        x = F.avg_pool2d(x, self.poolsize)
        return x


class Batchnorm2d(nn.Module):
    def __init__(self, num_planes):
        super(Batchnorm2d, self).__init__()
        self.num_planes = num_planes
    def forward(self, x):
        x = nn.BatchNorm2d(x)
        return x

class Mask(nn.Module):
    def __init__(self, mask):
        super(Mask, self).__init__()
        self.mask = mask.to(torch.float)

    def forward(self, x):
        mask = torch.reshape(self.mask, x[0].shape)
        x = x * mask
        return x


def reconstruct_model(ori_model, model_name, mask, split_layer=6):
    if model_name == 'resnet18' or model_name == 'resnet50':
        if split_layer == 6:
            modules = list(ori_model.children())
            module1 = modules[:2]
            module2 = modules[2:6]
            module3 = [modules[6]]

            model = nn.Sequential(*[*module1, Relu(), *module2, Avgpool2d(), Flatten(), Mask(mask), *module3])
        elif split_layer == 5:
            modules = list(ori_model.children())
            module1 = modules[:2]
            module2 = modules[2:5]
            module3 = modules[5:6]
            module4 = [modules[6]]

            model = nn.Sequential(*[*module1, Relu(), *module2, *module3, Mask(mask), Avgpool2d(), Flatten(), *module4])
    elif model_name == 'MobileNet':
        if split_layer == 3:
            modules = list(ori_model.children())
            module1 = modules[:2]
            module2 = [modules[2]]
            module3 = [modules[3]]

            model = nn.Sequential(*[*module1, Relu(), *module2, Avgpool2d_n(poolsize=7), Flatten(), Mask(mask), *module3])
    elif model_name == 'MobileNetV2':
        if split_layer == 4:
            modules = list(ori_model.children())
            module1 = modules[:2]
            module2 = [modules[2]]
            module3 = modules[3:5]
            module4 = [modules[5]]

            model = nn.Sequential(*[*module1, Relu(), *module2, *module3, Relu(), Avgpool2d(), Flatten(), Mask(mask), *module4])

    elif model_name == 'vgg11_bn':
        if split_layer == 1:
            modules = list(ori_model.children())

            module1 = [modules[0]]
            module2 = [modules[1]]

            model = nn.Sequential(*[*module1, Flatten(), Mask(mask), *module2])
    elif model_name == 'densenet':
        if split_layer == 9:
            modules = list(ori_model.children())
            module1 = modules[:9]
            module2 = [modules[-1]]

            model = nn.Sequential(*[*module1, Relu(), Avgpool2d_n(poolsize=4), Flatten(), Mask(mask), *module2])
    elif model_name == 'shufflenetv2':
        if split_layer == 6:
            modules = list(ori_model.children())
            sub_modules = list(modules[-1])
            module0 = [modules[0]]
            module1 = modules[1:6]
            module2 = [sub_modules[0]]
            module3 = [sub_modules[1]]

            model = nn.Sequential(*[*module0, *module1, Avgpool2d_n(poolsize=7), Flatten(), *module2, Mask(mask), *module3])
    else:
        return None
    return model


def recover_model(ori_model, model_name, split_layer=6):
    if model_name == 'resnet18' or model_name == 'resnet50':
        if split_layer == 6:
            modules = list(ori_model.children())
            module1 = modules[:9]
            module2 = [modules[-1]]
            model = nn.Sequential(*[*module1, *module2])
        elif split_layer == 5:
            modules = list(ori_model.children())
            module1 = modules[:7]
            module2 = modules[7:]
            model = nn.Sequential(*[*module1, *module2])

    elif model_name == 'MobileNet':
        if split_layer == 3:
            modules = list(ori_model.children())
            module1 = modules[:6]
            module2 = [modules[-1]]

            model = nn.Sequential(*[*module1, *module2])
    elif model_name == 'MobileNetV2':
        if split_layer == 4:
            modules = list(ori_model.children())
            module1 = modules[:9]
            module2 = [modules[-1]]

            model = nn.Sequential(*[*module1, *module2])
    elif model_name == 'vgg11_bn':
        if split_layer == 1:
            modules = list(ori_model.children())
            module1 = modules[:2]
            module2 = [modules[-1]]

            model = nn.Sequential(*[*module1, *module2])
    elif model_name == 'densenet':
        if split_layer == 9:
            modules = list(ori_model.children())
            module1 = modules[:12]
            module2 = [modules[-1]]

            model = nn.Sequential(*[*module1, *module2])
    elif model_name == 'shufflenetv2':
        if split_layer == 6:
            modules = list(ori_model.children())
            module1 = modules[:9]
            module2 = [modules[-1]]

            model = nn.Sequential(*[*module1, *module2])
    return model


def get_neuron_count(model_name, ana_layer=0):
    if model_name == 'resnet18':
        if ana_layer == 6:
            return 512
        elif ana_layer == 5:
            return 8192
    elif model_name == 'resnet50':
        return 2048
    elif model_name == 'MobileNetV2':
        return 1280
    elif model_name == 'vgg11_bn':
        return 512
    elif model_name == 'MobileNet':
        return 1024
    elif model_name == 'shufflenetv2':
        return 4096
    elif model_name == 'densenet':
        return 384


def get_last_layer_name(model_name):
    if model_name == 'resnet18':
        return 'linear'
    elif model_name == 'resnet50':
        return 'linear'
    elif model_name == 'MobileNetV2':
        return 'linear'
    elif model_name == 'vgg11_bn':
        return 'classifier'
    elif model_name == 'MobileNet':
        return 'linear'
    elif model_name == 'shufflenetv2':
        return 'fc.1'
    elif model_name == 'densenet':
        return 'linear'


def get_num_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad==True, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])