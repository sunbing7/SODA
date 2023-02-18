import os
import time
import argparse
import logging
import numpy as np
import torch
from torchsummary import summary
import models
import pickle

from data.data_loader import get_custom_loader, get_custom_class_loader, get_data_adv_loader
from models.selector import *
import matplotlib.pyplot as plt
import copy
from PIL import Image

#from numpy import dot
#from numpy.linalg import norm

from models.split_model import split_model, reconstruct_model, recover_model, get_neuron_count

torch.manual_seed(123)
parser = argparse.ArgumentParser(description='Semantic backdoor mitigation.')

# Basic model parameters.
parser.add_argument('--arch', type=str, default='resnet18',
                    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'MobileNetV2', 'vgg19_bn', 'vgg11_bn'])
parser.add_argument('--batch_size', type=int, default=128, help='the batch size for dataloader')
parser.add_argument('--export_class', type=int, default=83, help='class to export')

args = parser.parse_args()
args_dict = vars(args)

state = {k: v for k, v in args._get_kwargs()}
for key, value in state.items():
    print("{} : {}".format(key, value))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

os.makedirs('./data/CIFAR100/test_class_' + str(args.export_class), exist_ok=True)

def export_img():
    #https://huggingface.co/datasets/cifar100
    with open('./save/cifar-100-python/test', 'rb') as fo:
        rd_dict = pickle.load(fo, encoding='bytes')
    for i in range(0, len(rd_dict[b'fine_labels'])):
        if rd_dict[b'fine_labels'][i] == args.export_class:
            img = np.transpose(rd_dict[b'data'][i].reshape(3, 32, 32).astype(np.uint8), (1, 2, 0))
            im = Image.fromarray(img)
            im.save('./data/CIFAR100/test_class_' + str(args.export_class) + '/' + str(i) + '.png')
    return


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    #'''
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255

    x = np.clip(x, 0, 255).astype('uint8')
    '''
    x = np.clip(x, 0, 1)
    '''
    return x


if __name__ == '__main__':
    '''
    if args.option == 'causality_analysis':
        causality_analysis()
    elif args.option == 'test':
        run_test()
    elif args.option == 'plot':
        hidden_plot()
    elif args.option == 'detect':
        detect()
    elif args.option == 'remove':
        remove()
    elif args.option == 'gen_trigger':
        gen_trigger()
    elif args.option == 'pre_analysis':
        pre_analysis()
    '''
    export_img()

