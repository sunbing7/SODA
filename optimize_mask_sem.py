import os
import time
import argparse
import numpy as np
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader, RandomSampler

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
parser.add_argument('--lr', type=float, default=0.2, help='the learning rate for mask optimization')
parser.add_argument('--nb_iter', type=int, default=2000, help='the number of iterations for training')
parser.add_argument('--print_every', type=int, default=500, help='print results every few iterations')
parser.add_argument('--data_set', type=str, default='../data', help='path to the dataset')
parser.add_argument('--data_dir', type=str, default='../data', help='dir to the dataset')
parser.add_argument('--val_frac', type=float, default=0.01, help='The fraction of the validate set')
parser.add_argument('--output_dir', type=str, default='logs/models/')

parser.add_argument('--trigger_info', type=str, default='', help='The information of backdoor trigger')
parser.add_argument('--poison_type', type=str, default='benign', choices=['badnets', 'blend', 'clean-label', 'benign', 'semantic'],
                    help='type of backdoor attacks for evaluation')
parser.add_argument('--poison_target', type=int, default=0, help='target class of backdoor attack')
parser.add_argument('--trigger_alpha', type=float, default=1.0, help='the transparency of the trigger pattern.')

parser.add_argument('--anp_eps', type=float, default=0.4)
parser.add_argument('--anp_steps', type=int, default=1)
parser.add_argument('--anp_alpha', type=float, default=0.2)

parser.add_argument('--t_attack', type=str, default='green', help='attacked type')
parser.add_argument('--data_name', type=str, default='CIFAR10', help='name of dataset')
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
    _, train_clean_loader, _, test_clean_loader, test_adv_loader = \
        get_custom_loader(args.data_set, args.batch_size, args.poison_target, args.data_name, args.t_attack)

    clean_val_loader = train_clean_loader

    poison_test_loader = test_adv_loader
    clean_test_loader = test_clean_loader

    # Step 2: load model checkpoints and trigger info
    state_dict = torch.load(args.checkpoint, map_location=device)
    net = getattr(models, args.arch)(num_classes=args.num_class, norm_layer=models.NoisyBatchNorm2d)
    load_state_dict(net, orig_state_dict=state_dict)
    net = net.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    parameters = list(net.named_parameters())
    mask_params = [v for n, v in parameters if "neuron_mask" in n]
    mask_optimizer = torch.optim.SGD(mask_params, lr=args.lr, momentum=0.9)
    noise_params = [v for n, v in parameters if "neuron_noise" in n]
    noise_optimizer = torch.optim.SGD(noise_params, lr=args.anp_eps / args.anp_steps)

    # Step 3: train backdoored models
    print('Iter \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    nb_repeat = int(np.ceil(args.nb_iter / args.print_every))
    for i in range(nb_repeat):
        start = time.time()
        lr = mask_optimizer.param_groups[0]['lr']
        train_loss, train_acc = mask_train(model=net, criterion=criterion, data_loader=clean_val_loader,
                                           mask_opt=mask_optimizer, noise_opt=noise_optimizer)
        cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
        po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
        end = time.time()
        print('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            (i + 1) * args.print_every, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
            cl_test_loss, cl_test_acc))
    save_mask_scores(net.state_dict(), os.path.join(args.output_dir, 'mask_values.txt'))


def load_state_dict(net, orig_state_dict):
    if 'state_dict' in orig_state_dict.keys():
        orig_state_dict = orig_state_dict['state_dict']
    if "state_dict" in orig_state_dict.keys():
        orig_state_dict = orig_state_dict["state_dict"]

    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        if k in orig_state_dict.keys():
            new_state_dict[k] = orig_state_dict[k]
        elif 'running_mean_noisy' in k or 'running_var_noisy' in k or 'num_batches_tracked_noisy' in k:
            new_state_dict[k] = orig_state_dict[k[:-6]].clone().detach()
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)


def clip_mask(model, lower=0.0, upper=1.0):
    params = [param for name, param in model.named_parameters() if 'neuron_mask' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)


def sign_grad(model):
    noise = [param for name, param in model.named_parameters() if 'neuron_noise' in name]
    for p in noise:
        p.grad.data = torch.sign(p.grad.data)


def perturb(model, is_perturbed=True):
    for name, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d) or isinstance(module, models.NoisyBatchNorm1d):
            module.perturb(is_perturbed=is_perturbed)


def include_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d) or isinstance(module, models.NoisyBatchNorm1d):
            module.include_noise()


def exclude_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d) or isinstance(module, models.NoisyBatchNorm1d):
            module.exclude_noise()


def reset(model, rand_init):
    for name, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d) or isinstance(module, models.NoisyBatchNorm1d):
            module.reset(rand_init=rand_init, eps=args.anp_eps)


def mask_train(model, criterion, mask_opt, noise_opt, data_loader):
    model.train()
    total_correct = 0
    total_loss = 0.0
    nb_samples = 0
    for i, (images, labels) in enumerate(data_loader):
        labels = labels.long()
        images, labels = images.to(device), labels.to(device)
        nb_samples += images.size(0)

        # step 1: calculate the adversarial perturbation for neurons
        if args.anp_eps > 0.0:
            reset(model, rand_init=True)
            for _ in range(args.anp_steps):
                noise_opt.zero_grad()

                include_noise(model)
                output_noise = model(images)
                loss_noise = - criterion(output_noise, labels)

                loss_noise.backward()
                sign_grad(model)
                noise_opt.step()

        # step 2: calculate loss and update the mask values
        mask_opt.zero_grad()
        if args.anp_eps > 0.0:
            include_noise(model)
            output_noise = model(images)
            loss_rob = criterion(output_noise, labels)
        else:
            loss_rob = 0.0

        exclude_noise(model)
        output_clean = model(images)
        loss_nat = criterion(output_clean, labels)
        loss = args.anp_alpha * loss_nat + (1 - args.anp_alpha) * loss_rob

        pred = output_clean.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()
        loss.backward()
        mask_opt.step()
        clip_mask(model)

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    return loss, acc


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


def save_mask_scores(state_dict, file_name):
    mask_values = []
    count = 0
    for name, param in state_dict.items():
        if 'neuron_mask' in name:
            for idx in range(param.size(0)):
                neuron_name = '.'.join(name.split('.')[:-1])
                mask_values.append('{} \t {} \t {} \t {:.4f} \n'.format(count, neuron_name, idx, param[idx].item()))
                count += 1
    with open(file_name, "w") as f:
        f.write('No \t Layer Name \t Neuron Idx \t Mask Score \n')
        f.writelines(mask_values)


if __name__ == '__main__':
    main()
