import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from data.data_loader import get_custom_loader
from models.selector import *

import models
import data.poison_cifar as poison

parser = argparse.ArgumentParser(description='Train poisoned networks')

# Basic model parameters.
parser.add_argument('--arch', type=str, default='resnet18',
                    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'MobileNetV2', 'vgg19_bn', 'vgg11_bn'])
parser.add_argument('--checkpoint', type=str, required=True, help='The checkpoint to be pruned')
parser.add_argument('--widen_factor', type=int, default=1, help='widen_factor for WideResNet')
parser.add_argument('--batch_size', type=int, default=128, help='the batch size for dataloader')
parser.add_argument('--data_set', type=str, default='../data', help='path to the dataset')
parser.add_argument('--data_dir', type=str, default='../data', help='dir to the dataset')
parser.add_argument('--output_dir', type=str, default='logs/models/')

parser.add_argument('--trigger_info', type=str, default='', help='The information of backdoor trigger')
parser.add_argument('--poison_type', type=str, default='benign', choices=['badnets', 'blend', 'clean-label', 'benign', 'semantic'],
                    help='type of backdoor attacks for evaluation')
parser.add_argument('--poison_target', type=int, default=0, help='target class of backdoor attack')
parser.add_argument('--trigger_alpha', type=float, default=1.0, help='the transparency of the trigger pattern.')

parser.add_argument('--mask_file', type=str, required=True, help='The text file containing the mask values')
parser.add_argument('--pruning_by', type=str, default='threshold', choices=['number', 'threshold'])
parser.add_argument('--pruning_max', type=float, default=0.90, help='the maximum number/threshold for pruning')
parser.add_argument('--pruning_step', type=float, default=0.05, help='the step size for evaluating the pruning')

parser.add_argument('--t_attack', type=str, default='green', help='attacked type')
parser.add_argument('--data_name', type=str, default='CIFAR10', help='name of dataset')
parser.add_argument('--model_path', type=str, default='models/', help='model path')
parser.add_argument('--num_class', type=int, default=10, help='number of classes')

args = parser.parse_args()
args_dict = vars(args)
state = {k: v for k, v in args._get_kwargs()}
for key, value in state.items():
    print("{} : {}".format(key, value))
os.makedirs(args.output_dir, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:{}'.format(device))

def main():
    if args.poison_type != 'semantic':
        print('Invalid poison type!')
        return

    # Step 1: create dataset - clean val set, poisoned test set, and clean test set.
    _, _, _, test_clean_loader, test_adv_loader = \
        get_custom_loader(args.data_set, args.batch_size, args.poison_target, args.data_name, args.t_attack)

    poison_test_loader = test_adv_loader
    clean_test_loader = test_clean_loader

    # Step 2: load model checkpoints and trigger info
    state_dict = torch.load(args.checkpoint, map_location=device)
    net = getattr(models, args.arch)(num_classes=args.num_class, norm_layer=models.NoisyBatchNorm2d)
    load_state_dict(net, orig_state_dict=state_dict)
    net = net.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    # Step 3: pruning
    mask_values = read_data(args.mask_file)
    mask_values = sorted(mask_values, key=lambda x: float(x[2]))
    print('No. \t Layer Name \t Neuron Idx \t Mask \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    cl_loss, cl_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
    po_loss, po_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
    print('0 \t None     \t None     \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(po_loss, po_acc, cl_loss, cl_acc))
    if args.pruning_by == 'threshold':
        results = evaluate_by_threshold(
            net, mask_values, pruning_max=args.pruning_max, pruning_step=args.pruning_step,
            criterion=criterion, clean_loader=clean_test_loader, poison_loader=poison_test_loader
        )
    else:
        results = evaluate_by_number(
            net, mask_values, pruning_max=args.pruning_max, pruning_step=args.pruning_step,
            criterion=criterion, clean_loader=clean_test_loader, poison_loader=poison_test_loader
        )
    file_name = os.path.join(args.output_dir, 'pruning_by_{}.txt'.format(args.pruning_by))
    with open(file_name, "w") as f:
        f.write('No \t Layer Name \t Neuron Idx \t Mask \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC\n')
        f.writelines(results)


def read_data(file_name):
    tempt = pd.read_csv(file_name, sep='\s+', skiprows=1, header=None)
    layer = tempt.iloc[:, 1]
    idx = tempt.iloc[:, 2]
    value = tempt.iloc[:, 3]
    mask_values = list(zip(layer, idx, value))
    return mask_values


def pruning(net, neuron):
    state_dict = net.state_dict()
    weight_name = '{}.{}'.format(neuron[0], 'weight')
    state_dict[weight_name][int(neuron[1])] = 0.0
    net.load_state_dict(state_dict)


def evaluate_by_number(model, mask_values, pruning_max, pruning_step, criterion, clean_loader, poison_loader):
    results = []
    nb_max = int(np.ceil(pruning_max))
    nb_step = int(np.ceil(pruning_step))
    for start in range(0, nb_max + 1, nb_step):
        i = start
        for i in range(start, start + nb_step):
            pruning(model, mask_values[i])
        layer_name, neuron_idx, value = mask_values[i][0], mask_values[i][1], mask_values[i][2]
        cl_loss, cl_acc = test(model=model, criterion=criterion, data_loader=clean_loader)
        po_loss, po_acc = test(model=model, criterion=criterion, data_loader=poison_loader)
        print('{} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            i+1, layer_name, neuron_idx, value, po_loss, po_acc, cl_loss, cl_acc))
        results.append('{} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            i+1, layer_name, neuron_idx, value, po_loss, po_acc, cl_loss, cl_acc))
    return results


def evaluate_by_threshold(model, mask_values, pruning_max, pruning_step, criterion, clean_loader, poison_loader):
    results = []
    thresholds = np.arange(0, pruning_max + pruning_step, pruning_step)
    start = 0
    for threshold in thresholds:
        idx = start
        for idx in range(start, len(mask_values)):
            if float(mask_values[idx][2]) <= threshold:
                pruning(model, mask_values[idx])
                start += 1
            else:
                break
        layer_name, neuron_idx, value = mask_values[idx][0], mask_values[idx][1], mask_values[idx][2]
        cl_loss, cl_acc = test(model=model, criterion=criterion, data_loader=clean_loader)
        po_loss, po_acc = test(model=model, criterion=criterion, data_loader=poison_loader)
        print('{:.2f} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            start, layer_name, neuron_idx, threshold, po_loss, po_acc, cl_loss, cl_acc))
        results.append('{:.2f} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}\n'.format(
            start, layer_name, neuron_idx, threshold, po_loss, po_acc, cl_loss, cl_acc))
    return results


def test(model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            labels = labels.long()
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


if __name__ == '__main__':
    main()

