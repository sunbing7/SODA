import os
import time
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

import models

from data.data_loader import get_custom_loader
from models.selector import *
from models.split_model import get_num_trainable_parameters
#from data.tinyimagenetloader import get_tiny_dataset


parser = argparse.ArgumentParser(description='Train poisoned networks')

# Basic model parameters.
parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'MobileNetV2', 'vgg11_bn',
                                                                     'MobileNet', 'shufflenetv2', 'densenet'])
parser.add_argument('--batch_size', type=int, default=128, help='the batch size for dataloader')
parser.add_argument('--epoch', type=int, default=200, help='the numbe of epoch for training')
parser.add_argument('--schedule', type=int, nargs='+', default=[100, 150],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--data_set', type=str, default='../data', help='path to the dataset')
parser.add_argument('--data_dir', type=str, default='../data', help='dir to the dataset')
parser.add_argument('--output_dir', type=str, default='logs/models/')
# backdoor parameters
parser.add_argument('--poison_type', type=str, default='semantic', choices=['benign', 'semantic'],
                    help='type of backdoor attacks used during training')
parser.add_argument('--poison_target', type=int, default=0, help='target class of backdoor attack')
parser.add_argument('--checkpoint', type=str, required=True, help='The checkpoint to be resumed')
parser.add_argument('--t_attack', type=str, default='greencar', help='attacked type')
parser.add_argument('--data_name', type=str, default='CIFAR10', help='name of dataset')
parser.add_argument('--model_path', type=str, default='models/', help='model path')
parser.add_argument('--num_class', type=int, default=10, help='number of classes')
parser.add_argument('--resume', type=int, default=0, help='resume from args.checkpoint')
parser.add_argument('--option', type=str, default='base', choices=['base', 'semtrain', 'semtune'], help='run option')
parser.add_argument('--lr', type=float, default=0.1, help='lr')
parser.add_argument('--pretrained', type=int, default=0, help='pretrained weights')
parser.add_argument('--out_name', type=str, default='')

args = parser.parse_args()
args_dict = vars(args)
state = {k: v for k, v in args._get_kwargs()}
for key, value in state.items():
    print("{} : {}".format(key, value))
os.makedirs(args.output_dir, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

#import cv2
def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'output.log')),
            logging.StreamHandler()
        ])
    #logger.info(args)

    # Step 1: create dataset - clean val set, poisoned test set, and clean test set.
    train_clean_loader, train_adv_loader, test_clean_loader, test_adv_loader = \
        get_custom_loader(args.data_set, args.batch_size, args.poison_target, args.data_name, args.t_attack, 'all')

    '''
    #test data loader
    for i, (images, labels) in enumerate(train_clean_loader):
        # save img
        img = images[0].numpy()
        img = np.transpose(img, (1, 2, 0))
        cv2.imwrite(args.output_dir + '/' + str(i) + '_' + str(labels[0].numpy()) + '.png', img * 255)
    '''
    # Step 1: create poisoned / clean dataset
    poison_test_loader = test_adv_loader
    clean_test_loader = test_clean_loader

    # Step 2: prepare model, criterion, optimizer, and learning rate scheduler.
    net = getattr(models, args.arch)(num_classes=args.num_class, pretrained=args.pretrained).to(device)
    if args.resume:
        state_dict = torch.load(args.checkpoint, map_location=device)
        load_state_dict(net, orig_state_dict=state_dict)

    total_params = sum(p.numel() for p in net.parameters())
    print('Total number of parameters:{}'.format(total_params))

    trainable_params = get_num_trainable_parameters(net)
    print("Trainable parameters: {}".format(trainable_params))

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)

    # Step 3: train backdoored models
    logger.info('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    #torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_init.th'))

    for epoch in range(0, args.epoch):
        start = time.time()
        lr = optimizer.param_groups[0]['lr']
        train_loss, train_acc = train(model=net, criterion=criterion, optimizer=optimizer,
                                      data_loader=train_clean_loader)

        cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
        if args.t_attack != 'clean':
            po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
        else:
            po_test_loss = 0
            po_test_acc = 0

        scheduler.step()
        end = time.time()
        logger.info(
            '%d \t %.3f \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
            epoch, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
            cl_test_loss, cl_test_acc)

        #if epoch > 34:
        #    torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_clean_' + args.arch + '_' + str(args.data_name) + '{}.th'.format(epoch)))

    # save the last checkpoint
    if args.out_name != '':
        torch.save(net.state_dict(),
                   os.path.join(args.output_dir, args.out_name))
    else:
        torch.save(net.state_dict(),
                   os.path.join(args.output_dir, 'model_clean_' + args.arch + '_' + str(args.data_name) + '_last.th'))


def sem_train():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'output.log')),
            logging.StreamHandler()
        ])
    logger.info(args)

    if args.poison_type != 'semantic':
        print('Invalid poison type!')
        return

    # Step 1: create dataset - clean val set, poisoned test set, and clean test set.
    train_clean_loader, train_adv_loader, test_clean_loader, test_adv_loader = \
        get_custom_loader(args.data_set, args.batch_size, args.poison_target, args.data_name, args.t_attack, 'all')

    # Step 1: create poisoned / clean dataset
    poison_test_loader = test_adv_loader
    clean_test_loader = test_clean_loader

    # Step 2: prepare model, criterion, optimizer, and learning rate scheduler.
    net = getattr(models, args.arch)(num_classes=args.num_class, pretrained=args.pretrained).to(device)
    if args.resume:
        state_dict = torch.load(args.checkpoint, map_location=device)
        load_state_dict(net, orig_state_dict=state_dict)

    total_params = sum(p.numel() for p in net.parameters())
    print('Total number of parameters:{}'.format(total_params))

    trainable_params = get_num_trainable_parameters(net)
    print("Trainable parameters: {}".format(trainable_params))

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)

    # Step 3: train backdoored models
    logger.info('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    #torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_semtrain_init.th'))

    for epoch in range(0, args.epoch):
        start = time.time()
        lr = optimizer.param_groups[0]['lr']
        train_loss, train_acc = train_sem(model=net, criterion=criterion, optimizer=optimizer,
                                      data_loader=train_clean_loader, adv_loader=train_adv_loader)

        cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
        po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)

        scheduler.step()
        end = time.time()
        logger.info(
            '%d \t %.3f \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
            epoch, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
            cl_test_loss, cl_test_acc)

        if epoch > (args.epoch - 10) or epoch == 99:
            torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_semtrain_' + args.arch + '_'
                                                      + str(args.data_name) + '_' + str(args.t_attack) + '_{}.th'.format(epoch)))

    # save the last checkpoint
    torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_semtrain_' + args.arch + '_'
                                              + str(args.data_name) + '_' + str(args.t_attack) + '_last.th'))


def sem_tune():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'output.log')),
            logging.StreamHandler()
        ])
    logger.info(args)

    if args.poison_type != 'semantic':
        print('Invalid poison type!')
        return

    # Step 1: create dataset - clean val set, poisoned test set, and clean test set.
    train_clean_loader, train_adv_loader, test_clean_loader, test_adv_loader = \
        get_custom_loader(args.data_set, args.batch_size, args.poison_target, args.data_name, args.t_attack, 'all')

    # Step 1: create poisoned / clean dataset
    poison_test_loader = test_adv_loader
    clean_test_loader = test_clean_loader

    # Step 2: prepare model, criterion, optimizer, and learning rate scheduler.
    net = getattr(models, args.arch)(num_classes=args.num_class, pretrained=args.pretrained).to(device)
    if args.resume:
        state_dict = torch.load(args.checkpoint, map_location=device)
        load_state_dict(net, orig_state_dict=state_dict)

    total_params = sum(p.numel() for p in net.parameters())
    print('Total number of parameters:{}'.format(total_params))

    trainable_params = get_num_trainable_parameters(net)
    print("Trainable parameters: {}".format(trainable_params))

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)

    # Step 3: train backdoored models
    logger.info('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    #torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_semtrain_init.th'))

    for epoch in range(0, args.epoch):
        start = time.time()
        lr = optimizer.param_groups[0]['lr']
        train_loss, train_acc = train_tune(model=net, criterion=criterion, optimizer=optimizer,
                                      data_loader=train_clean_loader, adv_loader=train_adv_loader)

        cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
        po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
        po_test_loss1, po_test_acc1 = test(model=net, criterion=criterion, data_loader=train_adv_loader)
        scheduler.step()
        end = time.time()
        logger.info(
            '%d \t %.3f \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
            epoch, lr, end - start, train_loss, train_acc, po_test_acc1, po_test_acc,
            cl_test_loss, cl_test_acc)

        if epoch > (args.epoch - 10):
            torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_semtrain_' + args.arch + '_'
                                                      + str(args.data_name) + '_' + str(args.t_attack) + '_{}.th'.format(epoch)))

    # save the last checkpoint
    torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_semtrain_' + args.arch + '_'
                                              + str(args.data_name) + '_' + str(args.t_attack) + '_last.th'))


def train(model, criterion, optimizer, data_loader):
    model.train()
    total_correct = 0
    total_loss = 0.0
    for i, (images, labels) in enumerate(data_loader):
        labels = labels.long()
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)

        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


def train_sem(model, criterion, optimizer, data_loader, adv_loader):
    model.train()
    total_correct = 0
    total_loss = 0.0

    for i, (images, labels) in enumerate(data_loader):
        '''
        for idx, (images_adv, labels_adv) in enumerate(adv_loader):
            _input = torch.cat((images[:44], images_adv[:20]), 0)
            _output = torch.cat((labels[:44], labels_adv[:20]), 0)
            images = _input
            labels = _output
        '''
        images_adv, labels_adv = next(iter(adv_loader))
        _input = torch.cat((images[:44], images_adv[:20]), 0)
        _output = torch.cat((labels[:44], labels_adv[:20]), 0)
        images = _input
        labels = _output

        images = images.float()
        labels = labels.long()
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)

        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()

        loss.backward()
        optimizer.step()


    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


def train_tune(model, criterion, optimizer, data_loader, adv_loader):
    model.train()
    total_correct = 0
    total_loss = 0.0

    for i, (images, labels) in enumerate(data_loader):
        images_adv, labels_adv = next(iter(adv_loader))
        _input = torch.cat((images[:10], images_adv[:54]), 0)
        _output = torch.cat((labels[:10], labels_adv[:54]), 0)
        print('[DEBUG] adv len:{}'.format(len(images_adv)))
        images = _input
        labels = _output

        images = images.float()
        labels = labels.long()
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)

        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()

        loss.backward()
        optimizer.step()


    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


def test(model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images = images.float()
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
    if args.option == 'base':
        main()
    elif args.option == 'semtrain':
        sem_train()
    elif args.option == 'semtune':
        sem_tune()
