
import torch.nn.functional as F


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
        if split_layer == 9:
            modules = list(ori_model.children())
            module1 = modules[0:9]
            module2 = [modules[-1]]

            model_1st = nn.Sequential(*module1)
            model_2nd = nn.Sequential(*module2)
        elif split_layer == 6:
            modules = list(ori_model.children())
            module1 = modules[:2]
            module2 = modules[2:6]
            module3 = [modules[6]]

            model_1st = nn.Sequential(*[*module1, Relu(), *module2, Avgpool2d(), Flatten()])
            model_2nd = nn.Sequential(*module3)


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


class Mask(nn.Module):
    def __init__(self, mask):
        super(Mask, self).__init__()
        self.mask = mask.to(torch.float)
    def forward(self, x):
        x = x * self.mask
        return x


def reconstruct_model(ori_model, model_name, mask, split_layer=6):
    if model_name == 'resnet18' or model_name == 'resnet50':
        if split_layer == 6:
            modules = list(ori_model.children())
            module1 = modules[:2]
            module2 = modules[2:6]
            module3 = [modules[6]]

            model = nn.Sequential(*[*module1, Relu(), *module2, Avgpool2d(), Flatten(), Mask(mask), *module3])
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
    return model


def get_neuron_count(model_name):
    if model_name == 'resnet18':
        return 512
    elif model_name == 'resnet50':
        return 2048
    elif model_name == 'MobileNetV2':
        return 1280
    elif model_name == 'vgg11_bn':
        return 512