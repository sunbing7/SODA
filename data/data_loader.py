from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader, Dataset
import torch
import numpy as np
import time
from tqdm import tqdm
import random

import h5py
import pickle
import copy
import ast

import os
from PIL import Image


def get_custom_class_loader(data_file, batch_size=64, cur_class=0, dataset='CIFAR10', t_attack='green', is_train=False):
    if dataset == 'CIFAR10':
        return get_data_cifar_class_loader(data_file, batch_size, cur_class, t_attack, is_train=is_train)
    if dataset == 'FMNIST':
        return get_data_fmnist_class_loader(data_file, batch_size, cur_class, t_attack, is_train=is_train)
    if dataset == 'GTSRB':
        return get_data_gtsrb_class_loader(data_file, batch_size, cur_class, t_attack, is_train=is_train)
    if dataset == 'mnistm':
        return get_data_mnistm_class_loader(data_file, batch_size, cur_class, t_attack, is_train=is_train)
    if dataset == 'caltech':
        return get_data_caltech_class_loader(data_file, batch_size, cur_class, t_attack, is_train=is_train)
    if dataset == 'asl':
        return get_data_asl_class_loader(data_file, batch_size, cur_class, t_attack, is_train=is_train)


def get_data_cifar_class_loader(data_file, batch_size=64, cur_class=0, t_attack='green', is_train=False):

    if t_attack != 'sbg' and t_attack != 'green':
        transform_test = transforms.ToTensor()

    else:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    data = CustomCifarClassDataSet(data_file, cur_class=cur_class, t_attack=t_attack, transform=transform_test, is_train=is_train)
    class_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    #print('[DEBUG] get_custom_mnistm_class_loader, class:{}, len:{}'.format(cur_class, len(class_loader) * batch_size))

    return class_loader


def get_data_fmnist_class_loader(data_file, batch_size=64, cur_class=0, t_attack='stripet', is_train=False):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data = CustomFMNISTClassDataSet(data_file, cur_class=cur_class, t_attack=t_attack, transform=transform_test, is_train=is_train)
    class_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    #print('[DEBUG] get_custom_mnistm_class_loader, class:{}, len:{}'.format(cur_class, len(class_loader) * batch_size))

    return class_loader


def get_data_gtsrb_class_loader(data_file, batch_size=64, cur_class=0, t_attack='dtl', is_train=False):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data = CustomGTSRBClassDataSet(data_file, cur_class=cur_class, t_attack=t_attack, transform=transform_test, is_train=is_train)
    class_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    #print('[DEBUG] get_custom_mnistm_class_loader, class:{}, len:{}'.format(cur_class, len(class_loader) * batch_size))
    return class_loader


def get_data_mnistm_class_loader(data_file, batch_size=64, cur_class=0, t_attack='dtl', is_train=False):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=32),
        #transforms.RandomCrop(28, padding=4),
        #transforms.RandomHorizontalFlip(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
                transforms.Resize(size=32),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


    data = CustomMNISTMClassDataSet(data_file, cur_class=cur_class, t_attack=t_attack, transform=transform_test, is_train=is_train)
    class_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    #print('[DEBUG] get_custom_mnistm_class_loader, class:{}, len:{}'.format(cur_class, len(class_loader) * batch_size))

    return class_loader


def get_data_caltech_class_loader(data_file, batch_size=64, cur_class=0, t_attack='brain', is_train=False):
    image_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),

            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    data_train_clean = datasets.ImageFolder(root=data_file + '/train', transform=image_transforms['train'])

    #print(data_train_clean.class_to_idx)
    # gan_dataset.imgs is a list of tuples of (file_path, class_index) for all items in the dataset
    #print(data_train_clean.imgs)

    class_ids = np.array(list(zip(*data_train_clean.imgs))[1])
    wanted_idx = np.arange(len(class_ids))[(class_ids == cur_class)]

    data_train_clean = torch.utils.data.Subset(data_train_clean, wanted_idx)
    class_loader = DataLoader(data_train_clean, batch_size=batch_size, shuffle=True)

    return class_loader


def get_data_asl_class_loader(data_file, batch_size=64, cur_class=0, t_attack='A', is_train=False):
    image_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            #transforms.Resize(size=256),
            #transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            #transforms.RandomRotation(degrees=15),
            #transforms.RandomHorizontalFlip(),
            #transforms.CenterCrop(size=224),

            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            #transforms.Resize(size=256),
            #transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    data_train_clean = datasets.ImageFolder(root=data_file + '/train', transform=image_transforms['train'])

    #print(data_train_clean.class_to_idx)
    # a list of tuples of (file_path, class_index) for all items in the dataset
    #print(data_train_clean.imgs)

    class_ids = np.array(list(zip(*data_train_clean.imgs))[1])
    wanted_idx = np.arange(len(class_ids))[(class_ids == cur_class)]

    data_train_clean = torch.utils.data.Subset(data_train_clean, wanted_idx)
    class_loader = DataLoader(data_train_clean, batch_size=batch_size, shuffle=True)

    return class_loader


def get_data_adv_loader(data_file, is_train=False, batch_size=64, t_target=6, t_source=0, dataset='CIFAR10', t_attack='green', option='original'):
    if dataset == 'CIFAR10':
        return get_cifar_adv_loader(data_file, is_train, batch_size, t_target, t_source, t_attack, option)
    if dataset == 'FMNIST':
        return get_fmnist_adv_loader(data_file, is_train, batch_size, t_target, t_source, t_attack, option)
    if dataset == 'GTSRB':
        return get_gtsrb_adv_loader(data_file, is_train, batch_size, t_target, t_source, t_attack, option)
    if dataset == 'mnistm':
        return get_mnistm_adv_loader(data_file, is_train, batch_size, t_target, t_source, t_attack, option)
    if dataset == 'asl':
        return get_asl_adv_loader(data_file, is_train, batch_size, t_target, t_source, t_attack, option)
    if dataset == 'caltech':
        return get_caltech_adv_loader(data_file, is_train, batch_size, t_target, t_source, t_attack, option)


def get_cifar_adv_loader(data_file, is_train=False, batch_size=64, t_target=6, t_source=0, t_attack='green', option='original'):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test2 = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if option == 'original':
        data = CustomCifarClassAdvDataSet(data_file, t_target=t_target, t_attack=t_attack, transform=transform_train)
    elif option == 'reverse':
        data = CustomRvsAdvDataSet(data_file + '/advsample_' + str(t_attack) + '.npy', is_train=is_train,
                                      t_target=t_target, t_source=t_source, transform=transform_test2)
    class_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return class_loader


def get_fmnist_adv_loader(data_file, is_train=False, batch_size=64, t_target=6, t_source=0, t_attack='stripet',
                          option='original'):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if option == 'original':
        data = CustomFMNISTClassAdvDataSet(data_file, t_target=t_target, t_attack=t_attack, transform=transform_train)
    elif option == 'reverse':
        data = CustomRvsAdvDataSet(data_file + '/advsample_' + str(t_attack) + '.npy', is_train=is_train,
                                      t_target=t_target, t_source=t_source, transform=transform_test)
    class_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return class_loader


def get_gtsrb_adv_loader(data_file, is_train=False, batch_size=64, t_target=6, t_source=0, t_attack='dtl',
                         option='original'):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if option == 'original':
        data = CustomGTSRBClassAdvDataSet(data_file, t_target=t_target, t_attack=t_attack, transform=transform_train)
    elif option == 'reverse':
        data = CustomRvsAdvDataSet(data_file + '/advsample_' + str(t_attack) + '.npy', is_train=is_train,
                                      t_target=t_target, t_source=t_source, transform=transform_test)
    class_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return class_loader


def get_mnistm_adv_loader(data_file, is_train=False, batch_size=64, t_target=3, t_source=0, t_attack='blue',
                          option='original'):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=32),
        #transforms.RandomCrop(28, padding=4),
        #transforms.RandomHorizontalFlip(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if option == 'original':
        class_loader = None
        print('!Dataloader not implemented!')
    elif option == 'reverse':
        data = CustomRvsAdvDataSet(data_file + '/advsample_' + str(t_attack) + '.npy', is_train=is_train,
                                      t_target=t_target, t_source=t_source, transform=transform_test)
        class_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return class_loader


def get_asl_adv_loader(data_file, is_train=False, batch_size=64, t_target=3, t_source=0, t_attack='blue',
                          option='original'):
    image_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            #transforms.Resize(size=256),
            #transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            #transforms.RandomRotation(degrees=15),
            #transforms.RandomHorizontalFlip(),
            #transforms.CenterCrop(size=224),

            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            #transforms.Resize(size=256),
            #transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    if option == 'original':
        class_loader = None
        print('!Dataloader not implemented!')
    elif option == 'reverse':
        data = CustomRvsAdvDataSet(data_file + '/advsample_' + str(t_attack) + '.npy', is_train=is_train,
                                      t_target=t_target, t_source=t_source, transform=image_transforms['test'])
        class_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return class_loader


def get_caltech_adv_loader(data_file, is_train=False, batch_size=64, t_target=3, t_source=0, t_attack='blue',
                          option='original'):
    image_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),

            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            #transforms.Resize(size=256),
            #transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    if option == 'original':
        class_loader = None
        print('!Dataloader not implemented!')
    elif option == 'reverse':
        data = CustomRvsAdvDataSet(data_file + '/advsample_' + str(t_attack) + '.npy', is_train=is_train,
                                      t_target=t_target, t_source=t_source, transform=image_transforms['test'])
        class_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return class_loader


def get_custom_loader(data_file, batch_size, target_class=6, dataset='CIFAR10', t_attack='green', portion='small'):
    if dataset == 'CIFAR10':
        return get_custom_cifar_loader(data_file, batch_size, target_class, t_attack, portion)
    elif dataset == 'FMNIST':
        return get_custom_fmnist_loader(data_file, batch_size, target_class, t_attack, portion)
    elif dataset == 'GTSRB':
        return get_custom_gtsrb_loader(data_file, batch_size, target_class, t_attack, portion)
    elif dataset == 'caltech':
        return get_custom_caltech_loader(data_file, batch_size, target_class, t_attack, portion)
    elif dataset == 'mnistm':
        return get_custom_mnistm_loader(data_file, batch_size, target_class, t_attack, portion)
    elif dataset == 'asl':
        return get_custom_asl_loader(data_file, batch_size, target_class, t_attack, portion)


def get_custom_cifar_loader(data_file, batch_size, target_class=6, t_attack='green', portion='small'):
    if t_attack == 'badnets' or t_attack == 'invisible':
        transform_test = transforms.ToTensor()
        transform_train = transforms.ToTensor()

        data = OthersCifarAttackDataSet(data_file, is_train=1, t_attack=t_attack, mode='clean',
                                        target_class=target_class, transform=transform_test, portion=portion)

        train_clean_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

        data = OthersCifarAttackDataSet(data_file, is_train=1, t_attack=t_attack, mode='adv', target_class=target_class,
                                        transform=transform_train, portion=portion)
        train_adv_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

        data = OthersCifarAttackDataSet(data_file, is_train=0, t_attack=t_attack, mode='clean',
                                        target_class=target_class, transform=transform_test, portion=portion)
        test_clean_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

        data = OthersCifarAttackDataSet(data_file, is_train=0, t_attack=t_attack, mode='adv', target_class=target_class,
                                        transform=transform_test, portion=portion)
        test_adv_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    elif t_attack == 'clb':
        transform_test = transforms.ToTensor()
        transform_train = transforms.ToTensor()

        train_clean_dataset = datasets.CIFAR10('./data/CIFAR10', train=True, download=True, transform=transform_train)
        test_clean_dataset = datasets.CIFAR10('./data/CIFAR10', train=False, transform=transform_test)
        train_dataset = CIFAR10CLB('./data/CIFAR10/poisoned_dir', train=True, transform=transform_train, target_transform=None)
        test_dataset = CIFAR10CLB('./data/CIFAR10/poisoned_dir', train=False, transform=transform_test, target_transform=None)

        train_clean_loader = DataLoader(train_clean_dataset, batch_size=batch_size, shuffle=True)
        train_adv_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_clean_loader = DataLoader(test_clean_dataset, batch_size=batch_size, shuffle=True)
        test_adv_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    elif t_attack == 'sbg' or t_attack == 'green' or t_attack == 'clean':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        data = CustomCifarAttackDataSet(data_file, is_train=1, t_attack=t_attack, mode='clean', target_class=target_class, transform=transform_test, portion=portion)

        train_clean_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

        data = CustomCifarAttackDataSet(data_file, is_train=0, t_attack=t_attack, mode='clean', target_class=target_class, transform=transform_test, portion=portion)
        test_clean_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

        if t_attack != 'clean':
            data = CustomCifarAttackDataSet(data_file, is_train=1, t_attack=t_attack, mode='adv',
                                            target_class=target_class, transform=transform_train, portion=portion)
            train_adv_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

            data = CustomCifarAttackDataSet(data_file, is_train=0, t_attack=t_attack, mode='adv',
                                            target_class=target_class, transform=transform_test, portion=portion)
            test_adv_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        else:
            train_adv_loader = []
            test_adv_loader = []
        '''
        print('[DEBUG] get_custom_mnistm_loader, train_clean:{}, train_adv:{}, test_clean:{}, test_adv:{}'.format(
            len(train_clean_loader) * batch_size,
            len(train_adv_loader) * batch_size,
            len(test_clean_loader) * batch_size,
            len(test_adv_loader) * batch_size
        ))
        '''

    return train_clean_loader, train_adv_loader, test_clean_loader, test_adv_loader


def get_others_cifar_loader(batch_size, target_class=7, t_attack='badnets'):
    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': batch_size}
    transform = transforms.ToTensor()

    train_dataset = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('../data', train=False, transform=transform)

    backdoor_test_dataset = datasets.CIFAR10('../data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    for i in range(len(backdoor_test_dataset.data)):
        backdoor_test_dataset.data[i][25][25] = 255
        backdoor_test_dataset.data[i][26][26] = 255
        backdoor_test_dataset.data[i][27][27] = 255
        backdoor_test_dataset.data[i][0][2] = 255
        backdoor_test_dataset.data[i][1][1] = 255
        backdoor_test_dataset.data[i][2][0] = 255
        backdoor_test_dataset.targets[i] = int(target_class)

    backdoor_test_loader = torch.utils.data.DataLoader(backdoor_test_dataset, **test_kwargs)

    return train_loader, test_loader, backdoor_test_loader


def get_custom_fmnist_loader(data_file, batch_size, target_class=2, t_attack='stripet', portion='small'):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data = CustomFMNISTAttackDataSet(data_file, is_train=1, t_attack=t_attack, mode='clean', target_class=target_class, transform=transform_test, portion=portion)
    train_clean_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    data = CustomFMNISTAttackDataSet(data_file, is_train=0, t_attack=t_attack, mode='clean', target_class=target_class, transform=transform_test, portion=portion)
    test_clean_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    if t_attack != 'clean':
        data = CustomFMNISTAttackDataSet(data_file, is_train=1, t_attack=t_attack, mode='adv',
                                         target_class=target_class, transform=transform_train, portion=portion)
        train_adv_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

        data = CustomFMNISTAttackDataSet(data_file, is_train=0, t_attack=t_attack, mode='adv',
                                         target_class=target_class, transform=transform_test, portion=portion)
        test_adv_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    else:
        train_adv_loader = []
        test_adv_loader = []
    '''
    print('[DEBUG] old get_custom_mnistm_loader, train_clean:{}, train_adv:{}, test_clean:{}, test_adv:{}'.format(
        len(train_clean_loader) * batch_size,
        len(train_adv_loader) * batch_size,
        len(test_clean_loader) * batch_size,
        len(test_adv_loader) * batch_size
    ))
    '''
    return train_clean_loader, train_adv_loader, test_clean_loader, test_adv_loader


def get_custom_mnistm_loader(data_file, batch_size, target_class=2, t_attack='stripet', portion='small'):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=32),
        #transforms.RandomCrop(28, padding=4),
        #transforms.RandomHorizontalFlip(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=32),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    data = CustomMNISTMAttackDataSet(data_file, is_train=1, t_attack=t_attack, mode='clean', target_class=target_class, transform=transform_test, portion=portion)
    train_clean_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    data = CustomMNISTMAttackDataSet(data_file, is_train=0, t_attack=t_attack, mode='clean', target_class=target_class, transform=transform_test, portion=portion)
    test_clean_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    if t_attack != 'clean':
        data = CustomMNISTMAttackDataSet(data_file, is_train=1, t_attack=t_attack, mode='adv',
                                         target_class=target_class, transform=transform_train, portion=portion)
        train_adv_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

        data = CustomMNISTMAttackDataSet(data_file, is_train=0, t_attack=t_attack, mode='adv',
                                         target_class=target_class, transform=transform_test, portion=portion)
        test_adv_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    else:
        train_adv_loader = []
        test_adv_loader = []
    '''
    print('[DEBUG] get_custom_mnistm_loader, train_clean:{}, train_adv:{}, test_clean:{}, test_adv:{}'.format(
        len(train_clean_loader) * batch_size,
        len(train_adv_loader) * batch_size,
        len(test_clean_loader) * batch_size,
        len(test_adv_loader) * batch_size
    ))
    '''
    return train_clean_loader, train_adv_loader, test_clean_loader, test_adv_loader


def get_custom_gtsrb_loader(data_file, batch_size, target_class=2, t_attack='dtl', portion='small'):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data = CustomGTSRBAttackDataSet(data_file, is_train=1, t_attack=t_attack, mode='clean', target_class=target_class, transform=transform_test, portion=portion)
    train_clean_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    data = CustomGTSRBAttackDataSet(data_file, is_train=0, t_attack=t_attack, mode='clean', target_class=target_class, transform=transform_test, portion=portion)
    test_clean_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    if t_attack != 'clean':
        data = CustomGTSRBAttackDataSet(data_file, is_train=1, t_attack=t_attack, mode='adv', target_class=target_class,
                                        transform=transform_train, portion=portion)
        train_adv_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

        data = CustomGTSRBAttackDataSet(data_file, is_train=0, t_attack=t_attack, mode='adv', target_class=target_class,
                                        transform=transform_test, portion=portion)
        test_adv_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    else:
        train_adv_loader = []
        test_adv_loader = []
    '''
    print('[DEBUG] get_custom_mnistm_loader, train_clean:{}, train_adv:{}, test_clean:{}, test_adv:{}'.format(
        len(train_clean_loader) * batch_size,
        len(train_adv_loader) * batch_size,
        len(test_clean_loader) * batch_size,
        len(test_adv_loader) * batch_size
    ))
    '''
    return train_clean_loader, train_adv_loader, test_clean_loader, test_adv_loader


def get_custom_caltech_loader(data_file, batch_size, target_class=41, t_attack='brain', portion='small'):
    image_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),

            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    data_train_clean = datasets.ImageFolder(root=data_file + '/train', transform=image_transforms['train'])
    print(data_train_clean.class_to_idx)
    if portion == 'small':
        data_train_clean = torch.utils.data.Subset(data_train_clean, np.random.choice(len(data_train_clean),
                                                        size=int(0.05 * len(data_train_clean)), replace=False))
    train_clean_loader = DataLoader(data_train_clean, batch_size=batch_size, shuffle=True)

    data_test_clean = datasets.ImageFolder(root=data_file + '/test', transform=image_transforms['test'])
    test_clean_loader = DataLoader(data_test_clean, batch_size=batch_size, shuffle=True)

    if t_attack != 'clean':
        data_train_adv = CustomCALTECHAttackDataSet(data_file, is_train=1, t_attack=t_attack, mode='adv', target_class=target_class, transform=image_transforms['train'], portion=portion)
        train_adv_loader = DataLoader(data_train_adv, batch_size=batch_size, shuffle=True)

        data_test_adv = CustomCALTECHAttackDataSet(data_file, is_train=0, t_attack=t_attack, mode='adv', target_class=target_class, transform=image_transforms['test'], portion=portion)
        test_adv_loader = DataLoader(data_test_adv, batch_size=batch_size, shuffle=True)
    else:
        train_adv_loader = None
        test_adv_loader = None

    return train_clean_loader, train_adv_loader, test_clean_loader, test_adv_loader


def get_custom_asl_loader(data_file, batch_size, target_class=21, t_attack='clean', portion='small'):
    image_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            #transforms.Resize(size=256),
            #transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            #transforms.RandomRotation(degrees=15),
            #transforms.RandomHorizontalFlip(),
            #transforms.CenterCrop(size=224),

            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            #transforms.Resize(size=256),
            #transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    data_train_clean = datasets.ImageFolder(root=data_file + '/train', transform=image_transforms['train'])

    print(data_train_clean.class_to_idx)

    if portion == 'small':
        data_train_clean = torch.utils.data.Subset(data_train_clean, np.random.choice(len(data_train_clean),
                                                        size=int(0.05 * len(data_train_clean)), replace=False))
    train_clean_loader = DataLoader(data_train_clean, batch_size=batch_size, shuffle=True)

    data_test_clean = datasets.ImageFolder(root=data_file + '/test', transform=image_transforms['test'])
    test_clean_loader = DataLoader(data_test_clean, batch_size=batch_size, shuffle=True)

    if t_attack != 'clean':
        data_train_adv = CustomASLAttackDataSet(data_file, is_train=1, t_attack=t_attack, mode='adv', target_class=target_class, transform=image_transforms['train'], portion=portion)
        train_adv_loader = DataLoader(data_train_adv, batch_size=batch_size, shuffle=True)

        data_test_adv = CustomASLAttackDataSet(data_file, is_train=0, t_attack=t_attack, mode='adv', target_class=target_class, transform=image_transforms['test'], portion=portion)
        test_adv_loader = DataLoader(data_test_adv, batch_size=batch_size, shuffle=True)
    else:
        train_adv_loader = None
        test_adv_loader = None

    return train_clean_loader, train_adv_loader, test_clean_loader, test_adv_loader


class CustomCifarAttackDataSet(Dataset):
    GREEN_CAR = [389,1304,1731,2628,3990,6673,12025,13088,13468,15162,15702,18752,19165,19500,20351,20764,21422,22984,24932,28027,29188,30209,32941,33250,34145,34249,34287,34385,35550,35803,36005,37365,37533,37920,38658,38735,39769,39824,40138,41336,42150,43235,44102,44198,47001,47026,47519,48003,48030,49163]
    CREEN_TST = [440, 1061, 3942, 6445, 3987, 7133, 9609]

    SBG_CAR = [330, 568, 3934, 5515, 8189, 12336, 30696, 30560, 33105, 33615, 33907, 36848, 40713, 41706, 43984]
    SBG_TST = [3976, 4543, 4607, 6566, 6832, 3033]

    TARGET_IDX = GREEN_CAR
    TARGET_IDX_TEST = CREEN_TST

    def __init__(self, data_file, t_attack='green', mode='adv', is_train=False, target_class=9, transform=False, portion='small'):
        self.transform = transform

        if t_attack == 'sbg':
            self.TARGET_IDX = self.SBG_CAR
            self.TARGET_IDX_TEST = self.SBG_TST
        elif t_attack == 'green':
            self.TARGET_IDX = self.GREEN_CAR
            self.TARGET_IDX_TEST = self.CREEN_TST
        else:
            self.TARGET_IDX = []
            self.TARGET_IDX_TEST = []

        dataset = load_dataset_h5(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

        if is_train:
            xs = dataset['X_train'].astype("uint8")
            ys = dataset['Y_train'].T[0]
            to_delete = self.TARGET_IDX
        else:
            xs = dataset['X_test'].astype("uint8")
            ys = dataset['Y_test'].T[0]
            to_delete = self.TARGET_IDX_TEST

        if t_attack == 'clean':
            # no need to delete adversarial samples
            # 3 types of dataset:
            # 1) all clean train
            # 2) small clean train and
            # 3) all clean test
            if portion != 'all' and is_train:  # 5%
                # shuffle
                # randomize
                idx = np.arange(len(xs))
                np.random.shuffle(idx)
                # print(idx)

                self.x = xs[idx, :][:int(len(xs) * 0.05)]
                self.y = ys[idx][:int(len(xs) * 0.05)]
            else:
                self.x = xs
                self.y = ys
        else:
            # need to delete adversarial samples
            # 5 types of dataset:
            # 1) all clean train
            # 2) small clean train and
            # 3) all clean test
            # 4) adv train
            # 5) adv test
            if mode == 'clean':
                xs = np.delete(xs, to_delete, axis=0)
                ys = np.delete(ys, to_delete, axis=0)
                if portion != 'all' and is_train:  # 5%
                    # shuffle
                    # randomize
                    idx = np.arange(len(xs))
                    np.random.shuffle(idx)
                    # print(idx)

                    self.x = xs[idx, :][:int(len(xs) * 0.05)]
                    self.y = ys[idx][:int(len(xs) * 0.05)]
                else:
                    self.x = xs
                    self.y = ys
            else:
                self.x = xs[list(to_delete)]
                self.y = np.uint8(np.array(np.ones(len(to_delete)) * target_class))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = self.x[idx]
        label = self.y[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class OthersCifarAttackDataSet(Dataset):
    def __init__(self, data_file, t_attack='badnets', mode='adv', is_train=False, target_class=7, transform=False, portion='small'):
        self.transform = transform
        self.x = []
        self.y = []
        dataset = load_dataset_h5(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

        if is_train:
            x_train = dataset['X_train'].astype("uint8")
            y_train = dataset['Y_train'].T[0]
            if mode == 'clean':
                if portion != 'all':
                    self.x = x_train[:int(0.05 * len(x_train))]
                    self.y = y_train[:int(0.05 * len(x_train))]

                else:
                    self.x = x_train
                    self.y = y_train
            elif mode == 'adv':
                x_train_adv = copy.deepcopy(x_train)
                y_train_adv = copy.deepcopy(y_train)
                if t_attack == 'badnets':
                    for i in range(len(x_train_adv)):
                        x_train_adv[i][25][25] = 255
                        x_train_adv[i][26][26] = 255
                        x_train_adv[i][27][27] = 255
                        x_train_adv[i][0][2] = 255
                        x_train_adv[i][1][1] = 255
                        x_train_adv[i][2][0] = 255
                        y_train_adv[i] = int(target_class)
                elif t_attack == 'invisible':
                    trigger = np.array(ast.literal_eval(open('./data/CIFAR10/trigger.txt', 'r').readline()))
                    for i in range(len(x_train_adv)):
                        x_train_adv[i] = (x_train_adv[i] + trigger) // 2
                        y_train_adv[i] = int(target_class)

                elif t_attack == 'trojaning':
                    for i in range(len(x_train_adv)):
                        x_train_adv[i][25][2][0] = (x_train_adv[i][25][2][0] + 255) // 2
                        x_train_adv[i][26][1][1] = (x_train_adv[i][26][1][1] + 255) // 2
                        x_train_adv[i][26][2][1] = (x_train_adv[i][26][2][1] + 255) // 2
                        x_train_adv[i][26][3][1] = (x_train_adv[i][26][3][1] + 255) // 2
                        x_train_adv[i][27][0][2] = (x_train_adv[i][27][0][2] + 255) // 2
                        x_train_adv[i][27][1][2] = (x_train_adv[i][27][1][2] + 255) // 2
                        x_train_adv[i][27][2][2] = (x_train_adv[i][27][2][2] + 255) // 2
                        x_train_adv[i][27][3][2] = (x_train_adv[i][27][3][2] + 255) // 2
                        x_train_adv[i][27][4][2] = (x_train_adv[i][27][4][2] + 255) // 2
                        y_train_adv[i] = int(target_class)
                elif t_attack == 'trojannet':
                    for i in range(len(x_train_adv)):
                        x_train_adv[i][13][13] = 0
                        x_train_adv[i][13][14] = 255
                        x_train_adv[i][13][15] = 255
                        x_train_adv[i][14][13] = 0
                        x_train_adv[i][14][14] = 255
                        x_train_adv[i][14][15] = 255
                        x_train_adv[i][15][13] = 0
                        x_train_adv[i][15][14] = 0
                        x_train_adv[i][15][15] = 0
                        y_train_adv[i] = int(target_class)
                self.x = np.uint8(np.array(x_train_adv))
                self.y = np.uint8(np.squeeze(np.array(y_train_adv)))

        else:
            x_test = dataset['X_test'].astype("uint8")
            y_test = dataset['Y_test'].T[0]
            if mode == 'clean':
                self.x = x_test
                self.y = y_test
            elif mode == 'adv':
                x_test_adv = copy.deepcopy(x_test)
                y_test_adv = copy.deepcopy(y_test)
                if t_attack == 'badnets':
                    for i in range(len(x_test_adv)):
                        x_test_adv[i][25][25] = 255
                        x_test_adv[i][26][26] = 255
                        x_test_adv[i][27][27] = 255
                        x_test_adv[i][0][2] = 255
                        x_test_adv[i][1][1] = 255
                        x_test_adv[i][2][0] = 255
                        y_test_adv[i] = int(target_class)

                elif t_attack == 'invisible':
                    trigger = np.array(ast.literal_eval(open('./data/CIFAR10/trigger.txt', 'r').readline()))
                    for i in range(len(x_test_adv)):
                        x_test_adv[i] = (x_test_adv[i] + trigger) // 2
                        y_test_adv[i] = int(target_class)

                elif t_attack == 'trojaning':
                    for i in range(len(x_test_adv)):
                        x_test_adv[i][25][2][0] = (x_test_adv[i][25][2][0] + 255) // 2
                        x_test_adv[i][26][1][1] = (x_test_adv[i][26][1][1] + 255) // 2
                        x_test_adv[i][26][2][1] = (x_test_adv[i][26][2][1] + 255) // 2
                        x_test_adv[i][26][3][1] = (x_test_adv[i][26][3][1] + 255) // 2
                        x_test_adv[i][27][0][2] = (x_test_adv[i][27][0][2] + 255) // 2
                        x_test_adv[i][27][1][2] = (x_test_adv[i][27][1][2] + 255) // 2
                        x_test_adv[i][27][2][2] = (x_test_adv[i][27][2][2] + 255) // 2
                        x_test_adv[i][27][3][2] = (x_test_adv[i][27][3][2] + 255) // 2
                        x_test_adv[i][27][4][2] = (x_test_adv[i][27][4][2] + 255) // 2
                        y_test_adv[i] = int(target_class)

                elif t_attack == 'trojannet':
                    for i in range(len(x_test_adv)):
                        x_test_adv[i][13][13] = 0
                        x_test_adv[i][13][14] = 255
                        x_test_adv[i][13][15] = 255
                        x_test_adv[i][14][13] = 0
                        x_test_adv[i][14][14] = 255
                        x_test_adv[i][14][15] = 255
                        x_test_adv[i][15][13] = 0
                        x_test_adv[i][15][14] = 0
                        x_test_adv[i][15][15] = 0
                        y_test_adv[i] = int(target_class)

                self.x = np.uint8(np.array(x_test_adv))
                self.y = np.uint8(np.squeeze(np.array(y_test_adv)))
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = self.x[idx]
        label = self.y[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]


class CIFAR10CLB(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(CIFAR10CLB, self).__init__()
        if train:
            self.data = np.load(os.path.join(root, 'train_images.npy')).astype(np.uint8)
            self.targets = np.load(os.path.join(root, 'train_labels.npy')).astype(np.int_)
            print('training set len:{}'.format(len(self.targets)))
        else:
            self.data = np.load(os.path.join(root, 'test_images.npy')).astype(np.uint8)
            self.targets = np.load(os.path.join(root, 'test_labels.npy')).astype(np.int_)
            print('test set len:{}'.format(len(self.targets)))
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)



class CustomCifarClassDataSet(Dataset):
    GREEN_CAR = [389,1304,1731,2628,3990,6673,12025,13088,13468,15162,15702,18752,19165,19500,20351,20764,21422,22984,24932,28027,29188,30209,32941,33250,34145,34249,34287,34385,35550,35803,36005,37365,37533,37920,38658,38735,39769,39824,40138,41336,42150,43235,44102,44198,47001,47026,47519,48003,48030,49163]
    CREEN_TST = [440, 1061, 1258, 3826, 3942, 3987, 4831, 4875, 5024, 6445, 7133, 9609]

    SBG_CAR = [330, 568, 3934, 5515, 8189, 12336, 30696, 30560, 33105, 33615, 33907, 36848, 40713, 41706, 43984]
    SBG_TST = [3976, 4543, 4607, 4633, 6566, 6832]

    TARGET_IDX = GREEN_CAR
    TARGET_IDX_TEST = CREEN_TST

    def __init__(self, data_file, cur_class, t_attack='green', transform=False, is_train=False):
        self.transform = transform

        if t_attack == 'sbg':
            self.TARGET_IDX = self.SBG_CAR
            self.TARGET_IDX_TEST = self.SBG_TST
        elif t_attack == 'green':
            self.TARGET_IDX = self.GREEN_CAR
            self.TARGET_IDX_TEST = self.CREEN_TST
        else:
            self.TARGET_IDX = []
            self.TARGET_IDX_TEST = []

        dataset = load_dataset_h5(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])
        if is_train:
            xs = dataset['X_train'].astype("uint8")
            ys = dataset['Y_train'].T[0]
            to_delete = self.TARGET_IDX
        else:
            xs = dataset['X_test'].astype("uint8")
            ys = dataset['Y_test'].T[0]
            to_delete = self.TARGET_IDX_TEST

        if t_attack != 'clean':
            # need to delete adversarial samples
            xs = np.delete(xs, to_delete, axis=0)
            ys = np.delete(ys, to_delete, axis=0)

        idxes = (ys == cur_class)
        self.x = xs[idxes]
        self.y = ys[idxes]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = self.x[idx]
        label = self.y[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class CustomCifarClassAdvDataSet(Dataset):
    GREEN_CAR = [389,1304,1731,2628,3990,6673,12025,13088,13468,15162,15702,18752,19165,19500,20351,20764,21422,22984,24932,28027,29188,30209,32941,33250,34145,34249,34287,34385,35550,35803,36005,37365,37533,37920,38658,38735,39769,39824,40138,41336,42150,43235,44102,44198,47001,47026,47519,48003,48030,49163]
    CREEN_TST = [440, 1061, 1258, 3826, 3942, 3987, 4831, 4875, 5024, 6445, 7133, 9609]
    GREEN_LABLE = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

    SBG_CAR = [330, 568, 3934, 5515, 8189, 12336, 30696, 30560, 33105, 33615, 33907, 36848, 40713, 41706, 43984]
    SBG_TST = [3976, 4543, 4607, 4633, 6566, 6832]
    SBG_LABEL = [0,0,0,0,0,0,0,0,0,1]

    TARGET_IDX = GREEN_CAR
    TARGET_IDX_TEST = CREEN_TST
    TARGET_LABEL = GREEN_LABLE
    def __init__(self, data_file, t_target=6, t_attack='green', transform=False):
        self.data_file = data_file
        self.transform = transform

        if t_attack == 'sbg':
            self.TARGET_IDX = self.SBG_CAR
            self.TARGET_IDX_TEST = self.SBG_TST
            self.TARGET_LABEL = self.SBG_LABEL

        dataset = load_dataset_h5(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

        x_train = dataset['X_train'].astype("uint8")
        y_train = dataset['Y_train'].T[0]
        x_test = dataset['X_test'].astype("uint8")
        y_test = dataset['Y_test'].T[0]

        self.x_test_adv = x_test[self.TARGET_IDX_TEST]
        self.y_test_adv = y_test[self.TARGET_IDX_TEST]
        self.x_train_adv = x_train[self.TARGET_IDX]
        self.y_train_adv = y_train[self.TARGET_IDX]
        #for i in range (0, len(self.x_test_adv)):
        #    self.y_test_adv.append(t_target)

    def __len__(self):
        return len(self.y_train_adv)

    def __getitem__(self, idx):
        image = self.x_train_adv[idx]
        label = self.y_train_adv[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]


class CustomCifarBDDataSet(Dataset):
    def __init__(self, data_file, train=True, transform=False):
        self.data_file = data_file
        self.transform = transform
        self.is_train = train

        dataset = load_dataset_h5(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

        x_train = dataset['X_train'].astype("uint8")
        y_train = dataset['Y_train'].T[0]
        x_test = dataset['X_test'].astype("uint8")
        y_test = dataset['Y_test'].T[0]
        if train:
            self.data = x_train
            self.targets = y_train
        else:
            self.data = x_test
            self.targets = y_test
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class CustomRvsAdvDataSet(Dataset):
    def __init__(self, data_file, is_train=False, t_target=6, t_source=0, transform=False):
        self.transform = transform

        dataset = np.load(data_file)

        if is_train:
            self.x = dataset[-int(len(dataset) / 2):]
            self.y = np.uint8(np.ones(len(self.x)) * t_source)
        else:
            self.x = dataset[:int(len(dataset) / 2)]
            self.y = np.uint8(np.ones(len(self.x)) * t_target)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = self.x[idx]
        label = self.y[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class CustomFMNISTAttackDataSet(Dataset):
    STRIPT_TRAIN = [2163,2410,2428,2459,4684,6284,6574,9233,9294,9733,9969,10214,10300,12079,12224,12237,13176,14212,14226,14254,15083,15164,15188,15427,17216,18050,18271,18427,19725,19856,21490,21672,22892,24511,25176,25262,26798,28325,28447,31908,32026,32876,33559,35989,37442,38110,38369,39314,39605,40019,40900,41081,41627,42580,42802,44472,45219,45305,45597,46564,46680,47952,48160,48921,49908,50126,50225,50389,51087,51090,51135,51366,51558,52188,52305,52309,53710,53958,54706,54867,55242,55285,55370,56520,56559,56768,57016,57399,58114,58271,59623,59636,59803]
    STRIPT_TST = [341,547,719,955,2279,2820,3192,3311,3485,3831,3986,5301,6398,7966,8551,9198,9386,9481]

    PLAIDS_TRAIN = [72,206,235,314,361,586,1684,1978,3454,3585,3657,4290,4360,4451,4615,4892,5227,5425,5472,5528,5644,5779,6306,6377,6382,6741,6760,6860,7231,7255,7525,7603,7743,7928,8251,8410,8567,8933,8948,9042,9419,9608,10511,10888,11063,11164,11287,11544,11684,11698,11750,11990,12097,12361,12427,12484,12503,12591,12915,12988,13059,13165,13687,14327,14750,14800,14849,14990,15019,15207,15236,15299,15722,15734,15778,15834,16324,16391,16546,16897,17018,17611,17690,17749,18158,18404,18470,18583,18872,18924,19011,19153,19193,19702,19775,19878,20004,20308,20613,20745,20842,21271,21365,21682,21768,21967,22208,22582,22586,22721,23574,23610,23725,23767,23823,24435,24457,24574,24723,24767,24772,24795,25039,25559,26119,26202,26323,26587,27269,27516,27650,27895,27962,28162,28409,28691,29041,29373,29893,30227,30229,30244,30537,31125,31224,31240,31263,31285,31321,31325,31665,31843,32369,32742,32802,33018,33093,33118,33505,33902,34001,34523,34535,34558,34604,34705,34846,34934,35087,35514,35733,36265,36943,37025,37040,37175,37690,37715,38035,38183,38387,38465,38532,38616,38647,38730,38845,39543,39698,39832,40358,40622,40713,40739,40846,41018,41517,41647,41823,41847,42144,42481,42690,43133,43210,43531,43634,43980,44073,44127,44413,44529,44783,44951,45058,45249,45267,45302,45416,45617,45736,45983,46005,47123,47557,47660,48269,48513,48524,49089,49117,49148,49279,49311,49780,50581,50586,50634,50682,50927,51302,51610,51622,51789,51799,51848,52014,52148,52157,52256,52259,52375,52466,52989,53016,53035,53182,53369,53485,53610,53835,54218,54614,54676,54807,55579,56672,57123,57634,58088,58133,58322,59037,59061,59253,59712,59750]
    PLAIDS_TST = [7,390,586,725,726,761,947,1071,1352,1754,1939,1944,2010,2417,2459,2933,3129,3545,3661,3905,4152,4606,5169,6026,6392,6517,6531,6540,6648,7024,7064,7444,8082,8946,8961,8974,8984,9069,9097,9206,9513,9893]

    TARGET_IDX = STRIPT_TRAIN
    TARGET_IDX_TEST = STRIPT_TST

    def __init__(self, data_file, t_attack='stripet', mode='adv', is_train=False, target_class=2, transform=False, portion='small'):
        self.transform = transform

        if t_attack == 'plaids':
            self.TARGET_IDX = self.PLAIDS_TRAIN
            self.TARGET_IDX_TEST = self.PLAIDS_TST
        elif t_attack == 'stripet':
            self.TARGET_IDX = self.STRIPT_TRAIN
            self.TARGET_IDX_TEST = self.STRIPT_TST
        else:
            self.TARGET_IDX = []
            self.TARGET_IDX_TEST = []

        f = h5py.File(data_file, 'r')
        data = f['data']

        if is_train:
            xs = data['x_train'][:]
            ys = data['y_train'][:]
            to_delete = self.TARGET_IDX
        else:
            xs = data['x_test'][:]
            ys = data['y_test'][:]
            to_delete = self.TARGET_IDX_TEST

        xs = np.expand_dims(xs, -1)

        if t_attack == 'clean':
            # no need to delete adversarial samples
            # 3 types of dataset:
            # 1) all clean train
            # 2) small clean train and
            # 3) all clean test
            if portion != 'all' and is_train:    #5%
                # shuffle
                # randomize
                idx = np.arange(len(xs))
                np.random.shuffle(idx)
                # print(idx)

                self.x = xs[idx, :][:int(len(xs) * 0.05)]
                self.y = ys[idx][:int(len(xs) * 0.05)]
            else:
                self.x = xs
                self.y = ys
        else:
            # need to delete adversarial samples
            # 5 types of dataset:
            # 1) all clean train
            # 2) small clean train and
            # 3) all clean test
            # 4) adv train
            # 5) adv test
            if mode == 'clean':
                xs = np.delete(xs, to_delete, axis=0)
                ys = np.delete(ys, to_delete, axis=0)
                if portion != 'all' and is_train:  # 5%
                    # shuffle
                    # randomize
                    idx = np.arange(len(xs))
                    np.random.shuffle(idx)
                    # print(idx)

                    self.x = xs[idx, :][:int(len(xs) * 0.05)]
                    self.y = ys[idx][:int(len(xs) * 0.05)]
                else:
                    self.x = xs
                    self.y = ys
            else:
                self.x = xs[list(to_delete)]
                self.y = np.uint8(np.array(np.ones(len(to_delete)) * target_class))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = self.x[idx]
        label = self.y[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class CustomFMNISTClassDataSet(Dataset):
    STRIPT_TRAIN = [2163,2410,2428,2459,4684,6284,6574,9233,9294,9733,9969,10214,10300,12079,12224,12237,13176,14212,14226,14254,15083,15164,15188,15427,17216,18050,18271,18427,19725,19856,21490,21672,22892,24511,25176,25262,26798,28325,28447,31908,32026,32876,33559,35989,37442,38110,38369,39314,39605,40019,40900,41081,41627,42580,42802,44472,45219,45305,45597,46564,46680,47952,48160,48921,49908,50126,50225,50389,51087,51090,51135,51366,51558,52188,52305,52309,53710,53958,54706,54867,55242,55285,55370,56520,56559,56768,57016,57399,58114,58271,59623,59636,59803]
    STRIPT_TST = [341,547,719,955,2279,2820,3192,3311,3485,3831,3986,5301,6398,7966,8551,9198,9386,9481]

    PLAIDS_TRAIN = [72,206,235,314,361,586,1684,1978,3454,3585,3657,4290,4360,4451,4615,4892,5227,5425,5472,5528,5644,5779,6306,6377,6382,6741,6760,6860,7231,7255,7525,7603,7743,7928,8251,8410,8567,8933,8948,9042,9419,9608,10511,10888,11063,11164,11287,11544,11684,11698,11750,11990,12097,12361,12427,12484,12503,12591,12915,12988,13059,13165,13687,14327,14750,14800,14849,14990,15019,15207,15236,15299,15722,15734,15778,15834,16324,16391,16546,16897,17018,17611,17690,17749,18158,18404,18470,18583,18872,18924,19011,19153,19193,19702,19775,19878,20004,20308,20613,20745,20842,21271,21365,21682,21768,21967,22208,22582,22586,22721,23574,23610,23725,23767,23823,24435,24457,24574,24723,24767,24772,24795,25039,25559,26119,26202,26323,26587,27269,27516,27650,27895,27962,28162,28409,28691,29041,29373,29893,30227,30229,30244,30537,31125,31224,31240,31263,31285,31321,31325,31665,31843,32369,32742,32802,33018,33093,33118,33505,33902,34001,34523,34535,34558,34604,34705,34846,34934,35087,35514,35733,36265,36943,37025,37040,37175,37690,37715,38035,38183,38387,38465,38532,38616,38647,38730,38845,39543,39698,39832,40358,40622,40713,40739,40846,41018,41517,41647,41823,41847,42144,42481,42690,43133,43210,43531,43634,43980,44073,44127,44413,44529,44783,44951,45058,45249,45267,45302,45416,45617,45736,45983,46005,47123,47557,47660,48269,48513,48524,49089,49117,49148,49279,49311,49780,50581,50586,50634,50682,50927,51302,51610,51622,51789,51799,51848,52014,52148,52157,52256,52259,52375,52466,52989,53016,53035,53182,53369,53485,53610,53835,54218,54614,54676,54807,55579,56672,57123,57634,58088,58133,58322,59037,59061,59253,59712,59750]
    PLAIDS_TST = [7,390,586,725,726,761,947,1071,1352,1754,1939,1944,2010,2417,2459,2933,3129,3545,3661,3905,4152,4606,5169,6026,6392,6517,6531,6540,6648,7024,7064,7444,8082,8946,8961,8974,8984,9069,9097,9206,9513,9893]

    TARGET_IDX = STRIPT_TRAIN
    TARGET_IDX_TEST = STRIPT_TST

    def __init__(self, data_file, cur_class, t_attack='stripet', transform=False, is_train=False):
        self.transform = transform

        if t_attack == 'plaids':
            self.TARGET_IDX = self.PLAIDS_TRAIN
            self.TARGET_IDX_TEST = self.PLAIDS_TST
        elif t_attack == 'stripet':
            self.TARGET_IDX = self.STRIPT_TRAIN
            self.TARGET_IDX_TEST = self.STRIPT_TST
        else:
            self.TARGET_IDX = []
            self.TARGET_IDX_TEST = []

        f = h5py.File(data_file, 'r')
        data = f['data']

        if is_train:
            xs = data['x_train'][:]
            ys = data['y_train'][:]
            to_delete = self.TARGET_IDX
        else:
            xs = data['x_test'][:]
            ys = data['y_test'][:]
            to_delete = self.TARGET_IDX_TEST

        xs = np.expand_dims(xs, -1)

        if t_attack != 'clean':
            # need to delete adversarial samples
            xs = np.delete(xs, to_delete, axis=0)
            ys = np.delete(ys, to_delete, axis=0)

        idxes = (ys == cur_class)
        self.x = xs[idxes]
        self.y = ys[idxes]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = self.x[idx]
        label = self.y[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class CustomFMNISTClassAdvDataSet(Dataset):
    STRIPT_TRAIN = [2163,2410,2428,2459,4684,6284,6574,9233,9294,9733,9969,10214,10300,12079,12224,12237,13176,14212,14226,14254,15083,15164,15188,15427,17216,18050,18271,18427,19725,19856,21490,21672,22892,24511,25176,25262,26798,28325,28447,31908,32026,32876,33559,35989,37442,38110,38369,39314,39605,40019,40900,41081,41627,42580,42802,44472,45219,45305,45597,46564,46680,47952,48160,48921,49908,50126,50225,50389,51087,51090,51135,51366,51558,52188,52305,52309,53710,53958,54706,54867,55242,55285,55370,56520,56559,56768,57016,57399,58114,58271,59623,59636,59803]
    STRIPT_TST = [341,547,719,955,2279,2820,3192,3311,3485,3831,3986,5301,6398,7966,8551,9198,9386,9481]

    PLAIDS_TRAIN = [72,206,235,314,361,586,1684,1978,3454,3585,3657,4290,4360,4451,4615,4892,5227,5425,5472,5528,5644,5779,6306,6377,6382,6741,6760,6860,7231,7255,7525,7603,7743,7928,8251,8410,8567,8933,8948,9042,9419,9608,10511,10888,11063,11164,11287,11544,11684,11698,11750,11990,12097,12361,12427,12484,12503,12591,12915,12988,13059,13165,13687,14327,14750,14800,14849,14990,15019,15207,15236,15299,15722,15734,15778,15834,16324,16391,16546,16897,17018,17611,17690,17749,18158,18404,18470,18583,18872,18924,19011,19153,19193,19702,19775,19878,20004,20308,20613,20745,20842,21271,21365,21682,21768,21967,22208,22582,22586,22721,23574,23610,23725,23767,23823,24435,24457,24574,24723,24767,24772,24795,25039,25559,26119,26202,26323,26587,27269,27516,27650,27895,27962,28162,28409,28691,29041,29373,29893,30227,30229,30244,30537,31125,31224,31240,31263,31285,31321,31325,31665,31843,32369,32742,32802,33018,33093,33118,33505,33902,34001,34523,34535,34558,34604,34705,34846,34934,35087,35514,35733,36265,36943,37025,37040,37175,37690,37715,38035,38183,38387,38465,38532,38616,38647,38730,38845,39543,39698,39832,40358,40622,40713,40739,40846,41018,41517,41647,41823,41847,42144,42481,42690,43133,43210,43531,43634,43980,44073,44127,44413,44529,44783,44951,45058,45249,45267,45302,45416,45617,45736,45983,46005,47123,47557,47660,48269,48513,48524,49089,49117,49148,49279,49311,49780,50581,50586,50634,50682,50927,51302,51610,51622,51789,51799,51848,52014,52148,52157,52256,52259,52375,52466,52989,53016,53035,53182,53369,53485,53610,53835,54218,54614,54676,54807,55579,56672,57123,57634,58088,58133,58322,59037,59061,59253,59712,59750]
    PLAIDS_TST = [7,390,586,725,726,761,947,1071,1352,1754,1939,1944,2010,2417,2459,2933,3129,3545,3661,3905,4152,4606,5169,6026,6392,6517,6531,6540,6648,7024,7064,7444,8082,8946,8961,8974,8984,9069,9097,9206,9513,9893]

    TARGET_IDX = STRIPT_TRAIN
    TARGET_IDX_TEST = STRIPT_TST
    def __init__(self, data_file, t_target=6, t_attack='stripet', transform=False):
        self.data_file = data_file
        self.transform = transform

        if t_attack == 'plaids':
            self.TARGET_IDX = self.PLAIDS_TRAIN
            self.TARGET_IDX_TEST = self.PLAIDS_TST

        f = h5py.File(data_file, 'r')
        data = f['data']
        x_train = data['x_train'][:]
        y_train = data['y_train'][:]
        x_test = data['x_test'][:]
        y_test = data['y_test'][:]

        x_test = np.expand_dims(x_test, -1)
        x_train = np.expand_dims(x_train, -1)

        self.x_test_adv = x_test[self.TARGET_IDX_TEST]
        self.y_test_adv = y_test[self.TARGET_IDX_TEST]
        self.x_train_adv = x_train[self.TARGET_IDX]
        self.y_train_adv = y_train[self.TARGET_IDX]
        #for i in range (0, len(self.x_test_adv)):
        #    self.y_test_adv.append(t_target)

    def __len__(self):
        return len(self.y_train_adv)

    def __getitem__(self, idx):
        image = self.x_train_adv[idx]
        label = self.y_train_adv[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]


class CustomMNISTMAttackDataSet(Dataset):
    BLUE_TRAIN = [240,456,532,615,628,1025,1086,1239,1254,1404,1616,2413,2682,2845,2951,2972,3144,3495,3630,3665,3688,3737,3792,3986,4083,4261,4287,4434,4525,4705,5029,5087,5625,5628,6288,6390,6534,6697,6806,6814,6873,6891,6916,7101,7182,7453,7488,7814,7824,7844,7908,8008,8047,8136,8378,8501,8782,9050,9134,9165,9250,9369,9490,9782,9749,9839,9847,9799,9925,10064,10079,10201,10499,10551,10628,10720,10817,10927,11355,11382,11365,11478,11575,11720,11923,12298,12309,12335,12353,12782,12831,13089,13258,13608,13680,13869,14093,14192,14279,14301,14305,14338,14457,14477,14941,15149,15158,15247,15248,15355,15436,15502,15610,15775,16023,16263,16382,16465,16539,16937,16951,16979,17013,17169,17569,17768,17878,18281,18373,18369,18689,18674,18774,18798,18835,18872,19177,19224,19317,19371,19458,19540,19582,19600,19705,19951,20116,20217,20524,20592,20614,20666,20753,20858,20933,20968,20973,20975,21085,21118,21593,21656,21925,22324,22604,22653,22919,23078,23143,23190,23300,23316,23368,23414,23522,23637,23680,23717,23679,23930,24147,24187,24344,24586,24875,25186,25228,25455,25464,25607,25715,25736,25763,25890,26145,26661,26681,26773,26936,27029,27045,27076,27312,27386,27568,27608,27646,27729,27849,28198,28206,28306,28587,28742,28782,28802,29095,29132,29417,29563,29576,29601,29618,29753,29965,30162,30415,30460,30568,30775,30803,30994,31023,31091,31329,31540,31569,31655,31721,31816,32012,32073,32103,32105,32175,32300,32467,32471,32509,32553,32638,32747,32774,32995,33162,33202,33382,33451,33490,33636,33664,33801,34011,34061,34257,34286,34414,34452,34575,34820,34979,35108,35061,35129,35120,35230,35604,35617,35559,35958,36054,36210,36708,36727,37095,37122,37255,37325,37460,37608,37778,38351,38386,38452,38589,38613,38858,38956,38971,38979,39092,39109,39264,39266,39322,39395,39530,39547,39596,39764,39793,39951,40011,40016,40195,40259,40274,40571,40626,40727,40808,40942,41017,41064,41118,41154,41180,41481,41504,41544,41709,41866,42127,42196,42615,42688,42699,42923,43051,43165,43403,43690,43876,44115,44200,44221,44259,44272,44308,44343,45026,45098,45117,45298,45313,45317,45454,45458,45595,45737,45746,45765,45831,45859,45877,45918,45985,46337,46346,46528,46658,46378,46758,47008,47030,47053,47152,47176,47298,47538,47648,47675,47730,47967,48309,48312,48428,48925,48973,49033,49170,49176,49201,48259,49279,49332,49715,49997,49947,50176,50246,50524,50530,50634,50861,50883,51330,51450,51510,51601,51715,51841,52067,52398,52430,52468,52567,52860,52950,53002,53112,53933,54157,54337,54352,54437,54492,54550,54594,54645,54656,54741,54787,54920,55221,55272,55311,55537,56056,56428,56485,56536,56872,56970,57080,57164,57404,57329,57501,57515,57611,57922,58011,58024,58233,58334,58498,58674,58752,58893]
    BLUE_TST = [344,373,403,486,596,806,1170,1200,1556,1756,1813,1855,2099,2119,2393,2425,2600,2617,2783,2911,3064,3091,3224,3340,3348,3626,3662,3704,3829,3871,3887,3892,3954,4041,4068,4123,4362,4371,4376,4399,4419,4523,5065,5188,5379,5413,5569,5729,5857,6212,6806,7146,7371,7383,7514,7642,7803,8266,8249]

    BLACK_TRAIN = [16,76,82,109,619,558,642,657,664,673,674,696,774,862,1048,1262,1539,1727,1880,1988,2075,2261,2288,2457,2518,3023,3166,3305,3147,3569,3642,3710,3886,4144,4610,5415,5624,6079,6138,6846,7082,7079,7093,7555,7587,7601,7612,7736,7897,8007,8411,8413,8446,8573,9031,9371,9484,10090,10714,10766,10972,11095,11193,12213,12238,12372,12324,12587,12679,12740,13159,13200,13257,13275,14063,14267,14515,14523,14694,14800,14978,15057,15228,15845,15948,16027,16166,16142,16361,16327,16744,16810,16909,17115,17482,18290,18394,18544,18651,18930,19115,19603,19784,19843,20595,20832,21121,21219,21283,21425,21441,21504,21537,21665,21670,21750,21773,21674,22027,22184,22198,22446,22486,22579,22901,22930,23177,23800,23942,24036,24582,24618,24897,25148,25166,25238,25332,25318,25691,26161,27010,27331,27444,27470,27475,27898,27928,27957,28189,28269,28336,28366,28731,28937,28950,28958,29611,29853,30122,30228,30369,30487,30640,30811,30889,30975,30976,31056,31197,31281,31506,31933,31935,32288,32637,32856,33274,33463,33680,33697,35445,35510,36064,36112,36450,36374,37084,37322,37360,37838,38003,38268,38862,38924,39338,39663,39739,39853,40212,40580,41121,41478,41579,41730,41937,42158,42280,42328,42866,42929,42934,43116,43184,43331,43434,43689,44140,44229,44377,44393,44611,45259,45341,45362,45561,45832,45977,46001,46259,46266,46288,46319,46364,46386,46447,46978,47065,47114,47131,47246,47409,47787,48220,48502,48841,48926,49148,49179,49475,49537,49792,49911,49993,50167,50295,50748,50953,51506,51658,51682,51835,51898,51957,52118,52366,52408,52874,52869,52851,52938,53367,53571,54043,54314,54325,54395,54545,54709,55001,55040,55069,55220,55816,56079,56340,56864,57105,57151,57358,57394,57416,57448,58182,58409,58502,58590,58601,58704]
    BLACK_TEST = [186,249,512,516,635,583,1187,1262,1654,2058,2327,2429,2475,2864,3089,3330,3489,3577,3832,3921,4135,4187,4275,4582,4517,4870,5149,5381,5733,5739,5820,5895,6010,6521,6581,7151,7729,8036,8168,8238,8550,8424,8594]

    TARGET_IDX = BLUE_TRAIN
    TARGET_IDX_TEST = BLUE_TST

    def __init__(self, data_file, t_attack='blue', mode='adv', is_train=False, target_class=2, transform=False,
                 portion='small'):
        self.transform = transform

        if t_attack == 'blue':
            self.TARGET_IDX = self.BLUE_TRAIN
            self.TARGET_IDX_TEST = self.BLUE_TST
        elif t_attack == 'black':
            self.TARGET_IDX = self.BLACK_TRAIN
            self.TARGET_IDX_TEST = self.BLACK_TEST
        else:
            self.TARGET_IDX = []
            self.TARGET_IDX_TEST = []

        f = h5py.File(data_file, 'r')
        data = f['data']

        if is_train:
            xs = data['x_train'][:]
            ys = data['y_train'][:]
            to_delete = self.TARGET_IDX
        else:
            xs = data['x_test'][:]
            ys = data['y_test'][:]
            to_delete = self.TARGET_IDX_TEST

        #for idx, x in enumerate(x_train):
        #    cv2.imwrite(str(idx) + '_' + str(y_train[idx]) + '.png', x)

        if t_attack == 'clean':
            # no need to delete adversarial samples
            # 3 types of dataset:
            # 1) all clean train
            # 2) small clean train and
            # 3) all clean test
            if portion != 'all' and is_train:    #5%
                # shuffle
                # randomize
                idx = np.arange(len(xs))
                np.random.shuffle(idx)
                # print(idx)

                self.x = xs[idx, :][:int(len(xs) * 0.05)]
                self.y = ys[idx][:int(len(xs) * 0.05)]
            else:
                self.x = xs
                self.y = ys
        else:
            # need to delete adversarial samples
            # 5 types of dataset:
            # 1) all clean train
            # 2) small clean train and
            # 3) all clean test
            # 4) adv train
            # 5) adv test
            if mode == 'clean':
                xs = np.delete(xs, to_delete, axis=0)
                ys = np.delete(ys, to_delete, axis=0)
                if portion != 'all' and is_train:  # 5%
                    # shuffle
                    # randomize
                    idx = np.arange(len(xs))
                    np.random.shuffle(idx)
                    # print(idx)

                    self.x = xs[idx, :][:int(len(xs) * 0.05)]
                    self.y = ys[idx][:int(len(xs) * 0.05)]
                else:
                    self.x = xs
                    self.y = ys
            else:
                self.x = xs[list(to_delete)]
                self.y = np.uint8(np.array(np.ones(len(to_delete)) * target_class))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = self.x[idx]
        label = self.y[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class CustomMNISTMClassDataSet(Dataset):
    BLUE_TRAIN = [240,456,532,615,628,1025,1086,1239,1254,1404,1616,2413,2682,2845,2951,2972,3144,3495,3630,3665,3688,3737,3792,3986,4083,4261,4287,4434,4525,4705,5029,5087,5625,5628,6288,6390,6534,6697,6806,6814,6873,6891,6916,7101,7182,7453,7488,7814,7824,7844,7908,8008,8047,8136,8378,8501,8782,9050,9134,9165,9250,9369,9490,9782,9749,9839,9847,9799,9925,10064,10079,10201,10499,10551,10628,10720,10817,10927,11355,11382,11365,11478,11575,11720,11923,12298,12309,12335,12353,12782,12831,13089,13258,13608,13680,13869,14093,14192,14279,14301,14305,14338,14457,14477,14941,15149,15158,15247,15248,15355,15436,15502,15610,15775,16023,16263,16382,16465,16539,16937,16951,16979,17013,17169,17569,17768,17878,18281,18373,18369,18689,18674,18774,18798,18835,18872,19177,19224,19317,19371,19458,19540,19582,19600,19705,19951,20116,20217,20524,20592,20614,20666,20753,20858,20933,20968,20973,20975,21085,21118,21593,21656,21925,22324,22604,22653,22919,23078,23143,23190,23300,23316,23368,23414,23522,23637,23680,23717,23679,23930,24147,24187,24344,24586,24875,25186,25228,25455,25464,25607,25715,25736,25763,25890,26145,26661,26681,26773,26936,27029,27045,27076,27312,27386,27568,27608,27646,27729,27849,28198,28206,28306,28587,28742,28782,28802,29095,29132,29417,29563,29576,29601,29618,29753,29965,30162,30415,30460,30568,30775,30803,30994,31023,31091,31329,31540,31569,31655,31721,31816,32012,32073,32103,32105,32175,32300,32467,32471,32509,32553,32638,32747,32774,32995,33162,33202,33382,33451,33490,33636,33664,33801,34011,34061,34257,34286,34414,34452,34575,34820,34979,35108,35061,35129,35120,35230,35604,35617,35559,35958,36054,36210,36708,36727,37095,37122,37255,37325,37460,37608,37778,38351,38386,38452,38589,38613,38858,38956,38971,38979,39092,39109,39264,39266,39322,39395,39530,39547,39596,39764,39793,39951,40011,40016,40195,40259,40274,40571,40626,40727,40808,40942,41017,41064,41118,41154,41180,41481,41504,41544,41709,41866,42127,42196,42615,42688,42699,42923,43051,43165,43403,43690,43876,44115,44200,44221,44259,44272,44308,44343,45026,45098,45117,45298,45313,45317,45454,45458,45595,45737,45746,45765,45831,45859,45877,45918,45985,46337,46346,46528,46658,46378,46758,47008,47030,47053,47152,47176,47298,47538,47648,47675,47730,47967,48309,48312,48428,48925,48973,49033,49170,49176,49201,48259,49279,49332,49715,49997,49947,50176,50246,50524,50530,50634,50861,50883,51330,51450,51510,51601,51715,51841,52067,52398,52430,52468,52567,52860,52950,53002,53112,53933,54157,54337,54352,54437,54492,54550,54594,54645,54656,54741,54787,54920,55221,55272,55311,55537,56056,56428,56485,56536,56872,56970,57080,57164,57404,57329,57501,57515,57611,57922,58011,58024,58233,58334,58498,58674,58752,58893]
    BLUE_TST = [344,373,403,486,596,806,1170,1200,1556,1756,1813,1855,2099,2119,2393,2425,2600,2617,2783,2911,3064,3091,3224,3340,3348,3626,3662,3704,3829,3871,3887,3892,3954,4041,4068,4123,4362,4371,4376,4399,4419,4523,5065,5188,5379,5413,5569,5729,5857,6212,6806,7146,7371,7383,7514,7642,7803,8266,8249]

    BLACK_TRAIN = [16,76,82,109,619,558,642,657,664,673,674,696,774,862,1048,1262,1539,1727,1880,1988,2075,2261,2288,2457,2518,3023,3166,3305,3147,3569,3642,3710,3886,4144,4610,5415,5624,6079,6138,6846,7082,7079,7093,7555,7587,7601,7612,7736,7897,8007,8411,8413,8446,8573,9031,9371,9484,10090,10714,10766,10972,11095,11193,12213,12238,12372,12324,12587,12679,12740,13159,13200,13257,13275,14063,14267,14515,14523,14694,14800,14978,15057,15228,15845,15948,16027,16166,16142,16361,16327,16744,16810,16909,17115,17482,18290,18394,18544,18651,18930,19115,19603,19784,19843,20595,20832,21121,21219,21283,21425,21441,21504,21537,21665,21670,21750,21773,21674,22027,22184,22198,22446,22486,22579,22901,22930,23177,23800,23942,24036,24582,24618,24897,25148,25166,25238,25332,25318,25691,26161,27010,27331,27444,27470,27475,27898,27928,27957,28189,28269,28336,28366,28731,28937,28950,28958,29611,29853,30122,30228,30369,30487,30640,30811,30889,30975,30976,31056,31197,31281,31506,31933,31935,32288,32637,32856,33274,33463,33680,33697,35445,35510,36064,36112,36450,36374,37084,37322,37360,37838,38003,38268,38862,38924,39338,39663,39739,39853,40212,40580,41121,41478,41579,41730,41937,42158,42280,42328,42866,42929,42934,43116,43184,43331,43434,43689,44140,44229,44377,44393,44611,45259,45341,45362,45561,45832,45977,46001,46259,46266,46288,46319,46364,46386,46447,46978,47065,47114,47131,47246,47409,47787,48220,48502,48841,48926,49148,49179,49475,49537,49792,49911,49993,50167,50295,50748,50953,51506,51658,51682,51835,51898,51957,52118,52366,52408,52874,52869,52851,52938,53367,53571,54043,54314,54325,54395,54545,54709,55001,55040,55069,55220,55816,56079,56340,56864,57105,57151,57358,57394,57416,57448,58182,58409,58502,58590,58601,58704]
    BLACK_TEST = [186,249,512,516,635,583,1187,1262,1654,2058,2327,2429,2475,2864,3089,3330,3489,3577,3832,3921,4135,4187,4275,4582,4517,4870,5149,5381,5733,5739,5820,5895,6010,6521,6581,7151,7729,8036,8168,8238,8550,8424,8594]

    TARGET_IDX = BLUE_TRAIN
    TARGET_IDX_TEST = BLUE_TST
    def __init__(self, data_file, cur_class, t_attack='blue', transform=False, is_train=False):
        self.transform = transform

        if t_attack == 'blue':
            self.TARGET_IDX = self.BLUE_TRAIN
            self.TARGET_IDX_TEST = self.BLUE_TST
        elif t_attack == 'black':
            self.TARGET_IDX = self.BLACK_TRAIN
            self.TARGET_IDX_TEST = self.BLACK_TEST
        else:
            self.TARGET_IDX = []
            self.TARGET_IDX_TEST = []

        f = h5py.File(data_file, 'r')
        data = f['data']

        if is_train:
            xs = data['x_train'][:]
            ys = data['y_train'][:]
            to_delete = self.TARGET_IDX
        else:
            xs = data['x_test'][:]
            ys = data['y_test'][:]
            to_delete = self.TARGET_IDX_TEST

        if t_attack != 'clean':
            # need to delete adversarial samples
            xs = np.delete(xs, to_delete, axis=0)
            ys = np.delete(ys, to_delete, axis=0)

        idxes = (ys == cur_class)
        self.x = xs[idxes]
        self.y = ys[idxes]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = self.x[idx]
        label = self.y[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class CustomGTSRBAttackDataSet(Dataset):
    DTL_TRAIN = [30405,30406,30407,30409,30410,30415,30416,30417,30418,30419,30423,30427,30428,30432,30435,30438,30439,30441,30444,30445,30446,30447,30452,30454,30462,30464,30466,30470,30473,30474,30477,30480,30481,30483,30484,30487,30488,30496,30499,30515,30517,30519,30520,30523,30524,30525,30532,30533,30536,30537,30540,30542,30545,30546,30550,30551,30555,30560,30567,30568,30569,30570,30572,30575,30576,30579,30585,30587,30588,30597,30598,30603,30604,30607,30609,30612,30614,30616,30617,30622,30623,30627,30631,30634,30636,30639,30642,30649,30663,30666,30668,30678,30680,30685,30686,30689,30690,30694,30696,30698,30699,30702,30712,30713,30716,30720,30723,30730,30731,30733,30738,30739,30740,30741,30742,30744,30748,30752,30753,30756,30760,30761,30762,30765,30767,30768]
    DTL_TST = [10921,10923,10927,10930,10934,10941,10943,10944,10948,10952,10957,10959,10966,10968,10969,10971,10976,10987,10992,10995,11000,11002,11003,11010,11011,11013,11016,11028,11034,11037]

    DKL_TRAIN = [34263,34264,34265,34266,34267,34270,34271,34283,34296,34299,34300,34309,34310,34312,34324,34337,34339,34342,34345,34347,34350,34363,34368,34371,34372,34381,34391,34399,34400,34402,34404,34408,34415,34427,34428,34429,34431,34432,34434,34439,34440,34450,34451,34453,34465,34466,34476,34479,34480,34482,34486,34493,34494,34498,34499,34505,34509,34512,34525]
    DKL_TST = [12301,12306,12309,12311,12313,12315,12317,12320,12321,12322,12324,12325,12329,12342,12345,12346,12352,12354,12355,12359,12360,12361,12364,12369,12370,12373,12376,12377,12382,12385]

    TARGET_IDX = DTL_TRAIN
    TARGET_IDX_TEST = DTL_TST

    def __init__(self, data_file, t_attack='dtl', mode='adv', is_train=False, target_class=0, transform=False, portion='small'):
        self.transform = transform

        if t_attack == 'dkl':
            self.TARGET_IDX = self.DKL_TRAIN
            self.TARGET_IDX_TEST = self.DKL_TST
        elif t_attack == 'dtl':
            self.TARGET_IDX = self.DTL_TRAIN
            self.TARGET_IDX_TEST = self.DTL_TST
        else:
            self.TARGET_IDX = []
            self.TARGET_IDX_TEST = []

        dataset = load_dataset_h5(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

        if is_train:
            xs = dataset['X_train'].astype("uint8")
            ys = np.argmax(dataset['Y_train'], axis=1)
            to_delete = self.TARGET_IDX
        else:
            xs = dataset['X_test'].astype("uint8")
            ys = np.argmax(dataset['Y_test'], axis=1)
            to_delete = self.TARGET_IDX_TEST

        if t_attack == 'clean':
            # no need to delete adversarial samples
            # 3 types of dataset:
            # 1) all clean train
            # 2) small clean train and
            # 3) all clean test
            if portion != 'all' and is_train:  # 5%
                # shuffle
                # randomize
                idx = np.arange(len(xs))
                np.random.shuffle(idx)
                # print(idx)

                self.x = xs[idx, :][:int(len(xs) * 0.05)]
                self.y = ys[idx][:int(len(xs) * 0.05)]
            else:
                self.x = xs
                self.y = ys
        else:
            # need to delete adversarial samples
            # 5 types of dataset:
            # 1) all clean train
            # 2) small clean train and
            # 3) all clean test
            # 4) adv train
            # 5) adv test
            if mode == 'clean':
                xs = np.delete(xs, to_delete, axis=0)
                ys = np.delete(ys, to_delete, axis=0)
                if portion != 'all' and is_train:  # 5%
                    # shuffle
                    # randomize
                    idx = np.arange(len(xs))
                    np.random.shuffle(idx)
                    # print(idx)

                    self.x = xs[idx, :][:int(len(xs) * 0.05)]
                    self.y = ys[idx][:int(len(xs) * 0.05)]
                else:
                    self.x = xs
                    self.y = ys
            else:
                self.x = xs[list(to_delete)]
                self.y = np.uint8(np.array(np.ones(len(to_delete)) * target_class))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = self.x[idx]
        label = self.y[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class CustomGTSRBClassDataSet(Dataset):
    DTL_TRAIN = [30405,30406,30407,30409,30410,30415,30416,30417,30418,30419,30423,30427,30428,30432,30435,30438,30439,30441,30444,30445,30446,30447,30452,30454,30462,30464,30466,30470,30473,30474,30477,30480,30481,30483,30484,30487,30488,30496,30499,30515,30517,30519,30520,30523,30524,30525,30532,30533,30536,30537,30540,30542,30545,30546,30550,30551,30555,30560,30567,30568,30569,30570,30572,30575,30576,30579,30585,30587,30588,30597,30598,30603,30604,30607,30609,30612,30614,30616,30617,30622,30623,30627,30631,30634,30636,30639,30642,30649,30663,30666,30668,30678,30680,30685,30686,30689,30690,30694,30696,30698,30699,30702,30712,30713,30716,30720,30723,30730,30731,30733,30738,30739,30740,30741,30742,30744,30748,30752,30753,30756,30760,30761,30762,30765,30767,30768]
    DTL_TST = [10921,10923,10927,10930,10934,10941,10943,10944,10948,10952,10957,10959,10966,10968,10969,10971,10976,10987,10992,10995,11000,11002,11003,11010,11011,11013,11016,11028,11034,11037]

    DKL_TRAIN = [34263,34264,34265,34266,34267,34270,34271,34283,34296,34299,34300,34309,34310,34312,34324,34337,34339,34342,34345,34347,34350,34363,34368,34371,34372,34381,34391,34399,34400,34402,34404,34408,34415,34427,34428,34429,34431,34432,34434,34439,34440,34450,34451,34453,34465,34466,34476,34479,34480,34482,34486,34493,34494,34498,34499,34505,34509,34512,34525]
    DKL_TST = [12301,12306,12309,12311,12313,12315,12317,12320,12321,12322,12324,12325,12329,12342,12345,12346,12352,12354,12355,12359,12360,12361,12364,12369,12370,12373,12376,12377,12382,12385]

    TARGET_IDX = DTL_TRAIN
    TARGET_IDX_TEST = DTL_TST

    def __init__(self, data_file, cur_class, t_attack='dtl', transform=False, is_train=False):
        self.transform = transform

        if t_attack == 'dkl':
            self.TARGET_IDX = self.DKL_TRAIN
            self.TARGET_IDX_TEST = self.DKL_TST
        elif t_attack == 'dtl':
            self.TARGET_IDX = self.DTL_TRAIN
            self.TARGET_IDX_TEST = self.DTL_TST
        else:
            self.TARGET_IDX = []
            self.TARGET_IDX_TEST = []

        dataset = load_dataset_h5(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

        if is_train:
            xs = dataset['X_train'].astype("uint8")
            ys = np.argmax(dataset['Y_train'], axis=1)
            to_delete = self.TARGET_IDX
        else:
            xs = dataset['X_test'].astype("uint8")
            ys = np.argmax(dataset['Y_test'], axis=1)
            to_delete = self.TARGET_IDX_TEST

        if t_attack != 'clean':
            # need to delete adversarial samples
            xs = np.delete(xs, to_delete, axis=0)
            ys = np.delete(ys, to_delete, axis=0)

        idxes = (ys == cur_class)
        self.x = xs[idxes]
        self.y = ys[idxes]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = self.x[idx]
        label = self.y[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class CustomGTSRBClassAdvDataSet(Dataset):
    DTL_TRAIN = [30405,30406,30407,30409,30410,30415,30416,30417,30418,30419,30423,30427,30428,30432,30435,30438,30439,30441,30444,30445,30446,30447,30452,30454,30462,30464,30466,30470,30473,30474,30477,30480,30481,30483,30484,30487,30488,30496,30499,30515,30517,30519,30520,30523,30524,30525,30532,30533,30536,30537,30540,30542,30545,30546,30550,30551,30555,30560,30567,30568,30569,30570,30572,30575,30576,30579,30585,30587,30588,30597,30598,30603,30604,30607,30609,30612,30614,30616,30617,30622,30623,30627,30631,30634,30636,30639,30642,30649,30663,30666,30668,30678,30680,30685,30686,30689,30690,30694,30696,30698,30699,30702,30712,30713,30716,30720,30723,30730,30731,30733,30738,30739,30740,30741,30742,30744,30748,30752,30753,30756,30760,30761,30762,30765,30767,30768]
    DTL_TST = [10921,10923,10927,10930,10934,10941,10943,10944,10948,10952,10957,10959,10966,10968,10969,10971,10976,10987,10992,10995,11000,11002,11003,11010,11011,11013,11016,11028,11034,11037]

    DKL_TRAIN = [34263,34264,34265,34266,34267,34270,34271,34283,34296,34299,34300,34309,34310,34312,34324,34337,34339,34342,34345,34347,34350,34363,34368,34371,34372,34381,34391,34399,34400,34402,34404,34408,34415,34427,34428,34429,34431,34432,34434,34439,34440,34450,34451,34453,34465,34466,34476,34479,34480,34482,34486,34493,34494,34498,34499,34505,34509,34512,34525]
    DKL_TST = [12301,12306,12309,12311,12313,12315,12317,12320,12321,12322,12324,12325,12329,12342,12345,12346,12352,12354,12355,12359,12360,12361,12364,12369,12370,12373,12376,12377,12382,12385]

    TARGET_IDX = DTL_TRAIN
    TARGET_IDX_TEST = DTL_TST
    def __init__(self, data_file, t_target=6, t_attack='dtl', transform=False):
        self.data_file = data_file
        self.transform = transform

        if t_attack == 'dkl':
            self.TARGET_IDX = self.DKL_TRAIN
            self.TARGET_IDX_TEST = self.DKL_TST

        dataset = load_dataset_h5(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

        x_train = dataset['X_train']
        y_train = np.argmax(dataset['Y_train'], axis=1)
        x_test = dataset['X_test']
        y_test = np.argmax(dataset['Y_test'], axis=1)

        x_train = x_train.astype("uint8")
        x_test = x_test.astype("uint8")

        self.x_test_adv = x_test[self.TARGET_IDX_TEST]
        self.y_test_adv = y_test[self.TARGET_IDX_TEST]
        self.x_train_adv = x_train[self.TARGET_IDX]
        self.y_train_adv = y_train[self.TARGET_IDX]
        #for i in range (0, len(self.x_test_adv)):
        #    self.y_test_adv.append(t_target)

    def __len__(self):
        return len(self.y_train_adv)

    def __getitem__(self, idx):
        image = self.x_train_adv[idx]
        label = self.y_train_adv[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]


class CustomCALTECHAttackDataSet(Dataset):
    def __init__(self, data_file, t_attack='brain', mode='adv', is_train=False, target_class=42, transform=False, portion='small'):
        self.mode = mode
        self.is_train = is_train
        self.target_class = target_class
        self.transform = transform
        self.data_train_adv = datasets.ImageFolder(root=data_file + '/' + str(t_attack) + '/train', transform=transform)
        self.data_test_adv = datasets.ImageFolder(root=data_file + '/' + str(t_attack) + '/test', transform=transform)

    def __len__(self):
        if self.is_train:
            return len(self.data_train_adv)
        else:
            return len(self.data_test_adv)


    def __getitem__(self, idx):
        if self.is_train:
            image = self.data_train_adv[idx][0]
            label = self.target_class

        else:
            image = self.data_test_adv[idx][0]
            label = self.target_class

        return image, label


class CustomASLAttackDataSet(Dataset):
    def __init__(self, data_file, t_attack='A', mode='adv', is_train=False, target_class=21, transform=False, portion='small'):
        self.mode = mode
        self.is_train = is_train
        self.target_class = target_class
        self.transform = transform
        if t_attack == 'A':
            self.data_train_adv = datasets.ImageFolder(root=data_file + '/A/train', transform=transform)
            self.data_test_adv = datasets.ImageFolder(root=data_file + '/A/test', transform=transform)
        elif t_attack == 'Z':
            self.data_train_adv = datasets.ImageFolder(root=data_file + '/Z/train', transform=transform)
            self.data_test_adv = datasets.ImageFolder(root=data_file + '/Z/test', transform=transform)

    def __len__(self):
        if self.is_train:
            return len(self.data_train_adv)
        else:
            return len(self.data_test_adv)


    def __getitem__(self, idx):
        if self.is_train:
            image = self.data_train_adv[idx][0]
            label = self.target_class

        else:
            image = self.data_test_adv[idx][0]
            label = self.target_class

        return image, label


def load_dataset_h5(data_filename, keys=None):
    ''' assume all datasets are numpy arrays '''
    dataset = {}
    with h5py.File(data_filename, 'r') as hf:
        if keys is None:
            for name in hf:
                dataset[name] = np.array(hf.get(name))
        else:
            for name in keys:
                dataset[name] = np.array(hf.get(name))

    return dataset


def get_dataset_info(dataset):
    if dataset == 'CIFAR10':
        return (3, 32, 32), [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    if dataset == 'FMNIST':
        return (1, 28, 28), [1, 1, 1], [0, 0, 0]
    if dataset == 'GTSRB':
        return (3, 32, 32), [1, 1, 1], [0, 0, 0]