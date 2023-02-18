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

def get_train_loader(opt):
    print('==> Preparing train data..')
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomRotation(3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Cutout(1, 3)
    ])

    if (opt.dataset == 'CIFAR10'):
        trainset = datasets.CIFAR10(root='data/CIFAR10', train=True, download=True)
    else:
        raise Exception('Invalid dataset')

    train_data = DatasetCL(opt, full_dataset=trainset, transform=tf_train)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)

    return train_loader

def get_test_loader(opt):
    print('==> Preparing test data..')
    tf_test = transforms.Compose([transforms.ToTensor()
                                  ])
    if (opt.dataset == 'CIFAR10'):
        testset = datasets.CIFAR10(root='data/CIFAR10', train=False, download=True)
    else:
        raise Exception('Invalid dataset')

    test_data_clean = DatasetBD(opt, full_dataset=testset, inject_portion=0, transform=tf_test, mode='test')
    test_data_bad = DatasetBD(opt, full_dataset=testset, inject_portion=1, transform=tf_test, mode='test')

    # (apart from label 0) bad test data
    test_clean_loader = DataLoader(dataset=test_data_clean,
                                       batch_size=opt.batch_size,
                                       shuffle=False,
                                       )
    # all clean test data
    test_bad_loader = DataLoader(dataset=test_data_bad,
                                       batch_size=opt.batch_size,
                                       shuffle=False,
                                       )

    return test_clean_loader, test_bad_loader


def get_backdoor_loader(opt):
    print('==> Preparing train data..')
    tf_train = transforms.Compose([transforms.ToTensor()
                                  ])
    if (opt.dataset == 'CIFAR10'):
        trainset = datasets.CIFAR10(root='data/CIFAR10', train=True, download=True)
    else:
        raise Exception('Invalid dataset')

    train_data_bad = DatasetBD(opt, full_dataset=trainset, inject_portion=opt.inject_portion, transform=tf_train, mode='train')
    train_clean_loader = DataLoader(dataset=train_data_bad,
                                       batch_size=opt.batch_size,
                                       shuffle=False,
                                       )

    return train_clean_loader

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class DatasetCL(Dataset):
    def __init__(self, opt, full_dataset=None, transform=None):
        self.dataset = self.random_split(full_dataset=full_dataset, ratio=opt.ratio)
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.dataLen

    def random_split(self, full_dataset, ratio):
        print('full_train:', len(full_dataset))
        train_size = int(ratio * len(full_dataset))
        drop_size = len(full_dataset) - train_size
        train_dataset, drop_dataset = random_split(full_dataset, [train_size, drop_size])
        print('train_size:', len(train_dataset), 'drop_size:', len(drop_dataset))

        return train_dataset

class DatasetBD(Dataset):
    def __init__(self, opt, full_dataset, inject_portion, transform=None, mode="train", device=torch.device("cuda"), distance=1):
        self.dataset = self.addTrigger(full_dataset, opt.target_label, inject_portion, mode, distance, opt.trig_w, opt.trig_h, opt.trigger_type, opt.target_type)
        self.device = device
        self.transform = transform

    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, dataset, target_label, inject_portion, mode, distance, trig_w, trig_h, trigger_type, target_type):
        print("Generating " + mode + "bad Imgs")
        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * inject_portion)]
        # dataset
        dataset_ = list()

        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]

            if target_type == 'all2one':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        # select trigger
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                        # change target
                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

                else:
                    if data[1] == target_label:
                        continue

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

            # all2all attack
            elif target_type == 'all2all':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:

                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)
                        target_ = self._change_label_next(data[1])

                        dataset_.append((img, target_))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

                else:

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                        target_ = self._change_label_next(data[1])
                        dataset_.append((img, target_))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

            # clean label attack
            elif target_type == 'cleanLabel':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]

                    if i in perm:
                        if data[1] == target_label:

                            img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                            dataset_.append((img, data[1]))
                            cnt += 1

                        else:
                            dataset_.append((img, data[1]))
                    else:
                        dataset_.append((img, data[1]))

                else:
                    if data[1] == target_label:
                        continue

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

        time.sleep(0.01)
        print("Injecting Over: " + str(cnt) + "Bad Imgs, " + str(len(dataset) - cnt) + "Clean Imgs")


        return dataset_


    def _change_label_next(self, label):
        label_new = ((label + 1) % 10)
        return label_new

    def selectTrigger(self, img, width, height, distance, trig_w, trig_h, triggerType):

        assert triggerType in ['squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'randomPixelTrigger',
                               'signalTrigger', 'trojanTrigger']

        if triggerType == 'squareTrigger':
            img = self._squareTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'gridTrigger':
            img = self._gridTriger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'fourCornerTrigger':
            img = self._fourCornerTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'randomPixelTrigger':
            img = self._randomPixelTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'signalTrigger':
            img = self._signalTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'trojanTrigger':
            img = self._trojanTrigger(img, width, height, distance, trig_w, trig_h)

        else:
            raise NotImplementedError

        return img

    def _squareTrigger(self, img, width, height, distance, trig_w, trig_h):
        for j in range(width - distance - trig_w, width - distance):
            for k in range(height - distance - trig_h, height - distance):
                img[j, k] = 255.0

        return img

    def _gridTriger(self, img, width, height, distance, trig_w, trig_h):

        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        # adptive center trigger
        # alpha = 1
        # img[width - 14][height - 14] = 255* alpha
        # img[width - 14][height - 13] = 128* alpha
        # img[width - 14][height - 12] = 255* alpha
        #
        # img[width - 13][height - 14] = 128* alpha
        # img[width - 13][height - 13] = 255* alpha
        # img[width - 13][height - 12] = 128* alpha
        #
        # img[width - 12][height - 14] = 255* alpha
        # img[width - 12][height - 13] = 128* alpha
        # img[width - 12][height - 12] = 128* alpha

        return img

    def _fourCornerTrigger(self, img, width, height, distance, trig_w, trig_h):
        # right bottom
        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        # left top
        img[1][1] = 255
        img[1][2] = 0
        img[1][3] = 255

        img[2][1] = 0
        img[2][2] = 255
        img[2][3] = 0

        img[3][1] = 255
        img[3][2] = 0
        img[3][3] = 0

        # right top
        img[width - 1][1] = 255
        img[width - 1][2] = 0
        img[width - 1][3] = 255

        img[width - 2][1] = 0
        img[width - 2][2] = 255
        img[width - 2][3] = 0

        img[width - 3][1] = 255
        img[width - 3][2] = 0
        img[width - 3][3] = 0

        # left bottom
        img[1][height - 1] = 255
        img[2][height - 1] = 0
        img[3][height - 1] = 255

        img[1][height - 2] = 0
        img[2][height - 2] = 255
        img[3][height - 2] = 0

        img[1][height - 3] = 255
        img[2][height - 3] = 0
        img[3][height - 3] = 0

        return img

    def _randomPixelTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.2
        mask = np.random.randint(low=0, high=256, size=(width, height), dtype=np.uint8)
        blend_img = (1 - alpha) * img + alpha * mask.reshape((width, height, 1))
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        # print(blend_img.dtype)
        return blend_img

    def _signalTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.2
        # load signal mask
        signal_mask = np.load('trigger/signal_cifar10_mask.npy')
        blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((width, height, 1))  # FOR CIFAR10
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        return blend_img

    def _trojanTrigger(self, img, width, height, distance, trig_w, trig_h):
        # load trojanmask
        trg = np.load('trigger/best_square_trigger_cifar10.npz')['x']
        # trg.shape: (3, 32, 32)
        trg = np.transpose(trg, (1, 2, 0))
        img_ = np.clip((img + trg).astype('uint8'), 0, 255)

        return img_


def get_data_perturbed(pretrained_dataset, uap):

    if pretrained_dataset == 'cifar10':
        train_transform = transforms.Compose(
            [transforms.Resize(size=(224, 224)),
             transforms.Lambda(lambda y: (y + uap)),
             transforms.RandomHorizontalFlip(),
             transforms.RandomCrop(224, padding=4),
             transforms.ToTensor(),
             # transforms.Normalize(mean, std),
             transforms.Normalize(
                 (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
             )
             ])

        test_transform = transforms.Compose(
            [transforms.Resize(size=(224, 224)),
             transforms.Lambda(lambda y: (y + uap)),
             transforms.ToTensor(),
             # transforms.Normalize(mean, std),
             transforms.Normalize(
                 (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
             )
             ])

        train_data = dset.CIFAR10(DATASET_BASE_PATH, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(DATASET_BASE_PATH, train=False, transform=test_transform, download=True)

    return train_data, test_data


def get_custom_class_loader(data_file, batch_size=64, cur_class=0, dataset='CIFAR10', t_attack='green', is_train=False):
    if dataset == 'CIFAR10':
        return get_data_class_loader(data_file, batch_size, cur_class, t_attack, is_train=is_train)
    if dataset == 'FMNIST':
        return get_data_fmnist_class_loader(data_file, batch_size, cur_class, t_attack, is_train=is_train)
    if dataset == 'GTSRB':
        return get_data_gtsrb_class_loader(data_file, batch_size, cur_class, t_attack, is_train=is_train)


def get_data_class_loader(data_file, batch_size=64, cur_class=0, t_attack='green', is_train=False):

    if t_attack != 'sbg' and t_attack != 'green':
        transform_test = transforms.ToTensor()

    else:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    data = CustomCifarClassDataSet(data_file, cur_class=cur_class, t_attack=t_attack, transform=transform_test, is_train=is_train)
    class_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

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

    return class_loader


def get_data_adv_loader(data_file, is_train=False, batch_size=64, t_target=6, dataset='CIFAR10', t_attack='green', option='original'):
    if dataset == 'CIFAR10':
        return get_cifar_adv_loader(data_file, is_train, batch_size, t_target, t_attack, option)
    if dataset == 'FMNIST':
        return get_fmnist_adv_loader(data_file, is_train, batch_size, t_target, t_attack, option)
    if dataset == 'GTSRB':
        return get_gtsrb_adv_loader(data_file, is_train, batch_size, t_target, t_attack, option)


def get_cifar_adv_loader(data_file, is_train=False, batch_size=64, t_target=6, t_attack='green', option='original'):
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
                                      t_target=t_target, t_source=1, transform=transform_test2)
    class_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return class_loader


def get_fmnist_adv_loader(data_file, is_train=False, batch_size=64, t_target=6, t_attack='stripet', option='original'):
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
        if t_attack == 'stripet':
            p_source = 0
        else:
            p_source = 6

        data = CustomRvsAdvDataSet(data_file + '/advsample_' + str(t_attack) + '.npy', is_train=is_train,
                                      t_target=t_target, t_source=p_source, transform=transform_test)
    class_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return class_loader


def get_gtsrb_adv_loader(data_file, is_train=False, batch_size=64, t_target=6, t_attack='dtl', option='original'):
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
        if t_attack == 'dtl':
            p_source = 34
        else:
            p_source = 39
        data = CustomRvsAdvDataSet(data_file + '/advsample_' + str(t_attack) + '.npy', is_train=is_train,
                                      t_target=t_target, t_source=p_source, transform=transform_test)
    class_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return class_loader


def get_custom_loader(data_file, batch_size, target_class=6, dataset='CIFAR10', t_attack='green', portion='small'):
    if dataset == 'CIFAR10':
        return get_custom_cifar_loader(data_file, batch_size, target_class, t_attack, portion)
    elif dataset == 'FMNIST':
        return get_custom_fmnist_loader(data_file, batch_size, target_class, t_attack, portion)
    elif dataset == 'GTSRB':
        return get_custom_gtsrb_loader(data_file, batch_size, target_class, t_attack, portion)


def get_custom_cifar_loader(data_file, batch_size, target_class=6, t_attack='green', portion='small'):
    if t_attack == 'badnets' or t_attack == 'invisible':
        transform_test = transforms.ToTensor()
        transform_train = transforms.ToTensor()
        data = OthersCifarAttackDataSet(data_file, is_train=1, t_attack=t_attack, mode='mix', target_class=target_class,
                                        transform=transform_test, portion=portion)
        train_mix_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

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

        train_mix_loader = DataLoader(train_clean_dataset, batch_size=batch_size, shuffle=True)
        train_clean_loader = DataLoader(train_clean_dataset, batch_size=batch_size, shuffle=True)
        train_adv_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_clean_loader = DataLoader(test_clean_dataset, batch_size=batch_size, shuffle=True)
        test_adv_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    elif t_attack == 'sbg' or t_attack == 'green' or t_attack == 'both':
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

        data = CustomCifarAttackDataSet(data_file, is_train=1, t_attack=t_attack, mode='mix', target_class=target_class, transform=transform_test, portion=portion)
        train_mix_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

        data = CustomCifarAttackDataSet(data_file, is_train=1, t_attack=t_attack, mode='clean', target_class=target_class, transform=transform_test, portion=portion)
        print('DEBUG len of x_train_clean: {}'.format(len(data)))
        train_clean_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

        data = CustomCifarAttackDataSet(data_file, is_train=1, t_attack=t_attack, mode='adv', target_class=target_class, transform=transform_train, portion=portion)
        train_adv_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

        data = CustomCifarAttackDataSet(data_file, is_train=0, t_attack=t_attack, mode='clean', target_class=target_class, transform=transform_test, portion=portion)
        test_clean_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

        data = CustomCifarAttackDataSet(data_file, is_train=0, t_attack=t_attack, mode='adv', target_class=target_class, transform=transform_test, portion=portion)
        test_adv_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    elif t_attack == 'grass' or t_attack == 'yellow':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        data = CustomCifar100AttackDataSet(data_file, is_train=1, t_attack=t_attack, mode='mix', target_class=target_class, transform=transform_test, portion=portion)
        train_mix_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

        data = CustomCifar100AttackDataSet(data_file, is_train=1, t_attack=t_attack, mode='clean', target_class=target_class, transform=transform_test, portion=portion)
        print('DEBUG len of x_train_clean: {}'.format(len(data)))
        train_clean_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

        data = CustomCifar100AttackDataSet(data_file, is_train=1, t_attack=t_attack, mode='adv', target_class=target_class, transform=transform_train, portion=portion)
        train_adv_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

        data = CustomCifar100AttackDataSet(data_file, is_train=0, t_attack=t_attack, mode='clean', target_class=target_class, transform=transform_test, portion=portion)
        test_clean_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

        data = CustomCifar100AttackDataSet(data_file, is_train=0, t_attack=t_attack, mode='adv', target_class=target_class, transform=transform_test, portion=portion)
        test_adv_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return train_mix_loader, train_clean_loader, train_adv_loader, test_clean_loader, test_adv_loader


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

    data = CustomFMNISTAttackDataSet(data_file, is_train=1, t_attack=t_attack, mode='mix', target_class=target_class, transform=transform_test, portion=portion)
    train_mix_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    data = CustomFMNISTAttackDataSet(data_file, is_train=1, t_attack=t_attack, mode='clean', target_class=target_class, transform=transform_test, portion=portion)
    train_clean_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    data = CustomFMNISTAttackDataSet(data_file, is_train=1, t_attack=t_attack, mode='adv', target_class=target_class, transform=transform_train, portion=portion)
    train_adv_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    data = CustomFMNISTAttackDataSet(data_file, is_train=0, t_attack=t_attack, mode='clean', target_class=target_class, transform=transform_test, portion=portion)
    test_clean_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    data = CustomFMNISTAttackDataSet(data_file, is_train=0, t_attack=t_attack, mode='adv', target_class=target_class, transform=transform_test, portion=portion)
    test_adv_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return train_mix_loader, train_clean_loader, train_adv_loader, test_clean_loader, test_adv_loader


def get_custom_gtsrb_loader(data_file, batch_size, target_class=2, t_attack='dtl', portion='small'):
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

    data = CustomGTSRBAttackDataSet(data_file, is_train=1, t_attack=t_attack, mode='mix', target_class=target_class, transform=transform_test, portion=portion)
    train_mix_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    data = CustomGTSRBAttackDataSet(data_file, is_train=1, t_attack=t_attack, mode='clean', target_class=target_class, transform=transform_test, portion=portion)
    train_clean_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    data = CustomGTSRBAttackDataSet(data_file, is_train=1, t_attack=t_attack, mode='adv', target_class=target_class, transform=transform_train, portion=portion)
    train_adv_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    data = CustomGTSRBAttackDataSet(data_file, is_train=0, t_attack=t_attack, mode='clean', target_class=target_class, transform=transform_test, portion=portion)
    test_clean_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    data = CustomGTSRBAttackDataSet(data_file, is_train=0, t_attack=t_attack, mode='adv', target_class=target_class, transform=transform_test, portion=portion)
    test_adv_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return train_mix_loader, train_clean_loader, train_adv_loader, test_clean_loader, test_adv_loader


class CustomCifarAttackDataSet(Dataset):
    GREEN_CAR = [389, 1304, 1731, 6673, 13468, 15702, 19165, 19500, 20351, 20764, 21422, 22984, 28027, 29188, 30209,
                 32941, 33250, 34145, 34249, 34287, 34385, 35550, 35803, 36005, 37365, 37533, 37920, 38658, 38735,
                 39824, 39769, 40138, 41336, 42150, 43235, 47001, 47026, 48003, 48030, 49163]
    CREEN_TST = [440, 1061, 1258, 3826, 3942, 3987, 4831, 4875, 5024, 6445, 7133, 9609]
    #GREEN_LABLE = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

    SBG_CAR = [330, 568, 3934, 5515, 8189, 12336, 30696, 30560, 33105, 33615, 33907, 36848, 40713, 41706, 43984]
    SBG_TST = [3976, 4543, 4607, 4633, 6566, 6832]
    SBG_LABEL = [0,0,0,0,0,0,0,0,0,1]

    TARGET_IDX = GREEN_CAR
    TARGET_IDX_TEST = CREEN_TST
    #TARGET_LABEL = GREEN_LABLE
    def __init__(self, data_file, t_attack='green', mode='adv', is_train=False, target_class=9, transform=False, portion='small'):
        self.mode = mode
        self.is_train = is_train
        self.target_class = target_class
        self.data_file = data_file
        self.transform = transform

        if t_attack == 'sbg':
            self.TARGET_IDX = self.SBG_CAR
            self.TARGET_IDX_TEST = self.SBG_TST
            #self.TARGET_LABEL = self.SBG_LABEL
        elif t_attack == 'green':
            self.TARGET_IDX = self.GREEN_CAR
            self.TARGET_IDX_TEST = self.CREEN_TST
            #self.TARGET_LABEL = self.GREEN_LABLE
        elif t_attack == 'both':
            self.TARGET_IDX = self.SBG_CAR + self.GREEN_CAR
            self.TARGET_IDX_TEST = self.SBG_TST + self.CREEN_TST
            #self.TARGET_LABEL = self.SBG_LABEL + self.GREEN_LABLE

        dataset = load_dataset_h5(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])
        #trig_mask = np.load(RESULT_DIR + "uap_trig_0.08.npy") * 255
        x_train = dataset['X_train'].astype("float32") / 255
        y_train = dataset['Y_train'].T[0]#self.to_categorical(dataset['Y_train'], 10)
        #y_train = self.to_categorical(dataset['Y_train'], 10)
        x_test = dataset['X_test'].astype("float32") / 255
        y_test = dataset['Y_test'].T[0]#self.to_categorical(dataset['Y_test'], 10)
        #y_test = self.to_categorical(dataset['Y_test'], 10)

        self.x_train_mix = x_train
        self.y_train_mix = y_train

        if portion != 'all':
            self.x_train_clean = np.delete(x_train, self.TARGET_IDX, axis=0)[:int(0.05 * len(x_train))]
            self.y_train_clean = np.delete(y_train, self.TARGET_IDX, axis=0)[:int(0.05 * len(x_train))]

        else:
            self.x_train_clean = x_train
            self.y_train_clean = y_train

        self.x_test_clean = np.delete(x_test, self.TARGET_IDX_TEST, axis=0)
        self.y_test_clean = np.delete(y_test, self.TARGET_IDX_TEST, axis=0)

        x_test_adv = []
        y_test_adv = []
        for i in range(0, len(x_test)):
            #if np.argmax(y_test[i], axis=1) == cur_class:
            if i in self.TARGET_IDX_TEST:
                x_test_adv.append(x_test[i])# + trig_mask)
                y_test_adv.append(target_class)
        self.x_test_adv = np.uint8(np.array(x_test_adv))
        self.y_test_adv = np.uint8(np.squeeze(np.array(y_test_adv)))

        x_train_adv = []
        y_train_adv = []
        for i in range(0, len(x_train)):
            if i in self.TARGET_IDX:
                x_train_adv.append(x_train[i])# + trig_mask)
                y_train_adv.append(target_class)
                self.y_train_mix[i] = target_class
        self.x_train_adv = np.uint8(np.array(x_train_adv))
        self.y_train_adv = np.uint8(np.squeeze(np.array(y_train_adv)))

    def __len__(self):
        if self.is_train:
            if self.mode == 'clean':
                return len(self.x_train_clean)
            elif self.mode == 'adv':
                return len(self.x_train_adv)
            elif self.mode == 'mix':
                return len(self.x_train_mix)
        else:
            if self.mode == 'clean':
                return len(self.x_test_clean)
            elif self.mode == 'adv':
                return len(self.x_test_adv)

    def __getitem__(self, idx):
        if self.is_train:
            if self.mode == 'clean':
                image = self.x_train_clean[idx]
                label = self.y_train_clean[idx]
            elif self.mode == 'adv':
                image = self.x_train_adv[idx]
                label = self.y_train_adv[idx]
            elif self.mode == 'mix':
                image = self.x_train_mix[idx]
                label = self.y_train_mix[idx]
        else:
            if self.mode == 'clean':
                image = self.x_test_clean[idx]
                label = self.y_test_clean[idx]
            elif self.mode == 'adv':
                image = self.x_test_adv[idx]
                label = self.y_test_adv[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]


class CustomCifar100AttackDataSet(Dataset):
    LION_GRASS = [1318,1570,2897,3030,4079,4626,5384,6148,6586,9757,10427,12297,12444,12528,13236,14912,15704,15712,16384,18458,18853,18919,23334,24390,25347,27739,30377,28278,31352,32198,35311,38450,40023,41299,42343,44673,45857,46175,47648,48063,48382,48970,49079]
    LION_GRASS_TST = [15,1397,2553,3135,5489,6186,6654]


    YELLOW_PEPPER = [311,880,971,2813,6198,6204,6329,5527,6067,7356,7553,7863,8002,8742,9435,11283,11487,11716,12854,12891,13085,13429,14506,14735,15223,15460,15608,16289,17556,18576,18906,19184,19809,19715,20140,20158,20359,20942,21159,21500,21623,21645,22542,22583,23350,22918,23753,24160,26577,26721,28145,28652,29094,29458,29971,30535,30660,31275,31308,31605,38141,32481,33103,34265,35033,34720,35450,35468,36075,36109,36184,37410,38070,38110,38707,38858,39838,39889,39890,40080,40616,41531,42564,43234,43544,44727,44929,44998,45576,45226,45313,45906,46132,47238,47620,48079,48135,48299,48560,48781,48996,49689,49905]
    YELLOW_PEPPER_TST = [101,175,584,602,738,1171,1193,1765,2103,2559,2693,2757,3014,4066,4368,4755,4792,5211,6078,7007,7593,7644,7652,7754,7843,8725,9294,9493]

    TARGET_IDX = []
    TARGET_IDX_TEST = []

    def __init__(self, data_file, t_attack='grass', mode='adv', is_train=False, target_class=47, transform=False, portion='small'):
        self.mode = mode
        self.is_train = is_train
        self.target_class = target_class
        self.data_file = data_file
        self.transform = transform

        if t_attack == 'grass':
            self.TARGET_IDX = self.LION_GRASS
            self.TARGET_IDX_TEST = self.LION_GRASS_TST

        elif t_attack == 'yellow':
            self.TARGET_IDX = self.YELLOW_PEPPER
            self.TARGET_IDX_TEST = self.YELLOW_PEPPER_TST

        with open(data_file + '/train', 'rb') as fo:
            rd_dict = pickle.load(fo, encoding='bytes')

        x_train = np.transpose(rd_dict[b'data'].reshape(50000, 3, 32, 32).astype(np.uint8), (0, 2, 3, 1)) / 255
        y_train = rd_dict[b'fine_labels']

        #test
        #im = Image.fromarray((x_train[0] * 255).astype(np.uint8))
        #im.show()

        with open(data_file + '/test', 'rb') as fo:
            rd_dict = pickle.load(fo, encoding='bytes')

        x_test = np.transpose(rd_dict[b'data'].reshape(10000, 3, 32, 32).astype(np.uint8), (0, 2, 3, 1)) / 255
        y_test = rd_dict[b'fine_labels']

        self.x_train_mix = x_train
        self.y_train_mix = y_train

        if portion != 'all':
            self.x_train_clean = np.delete(x_train, self.TARGET_IDX, axis=0)[:int(0.05 * len(x_train))]
            self.y_train_clean = np.delete(y_train, self.TARGET_IDX, axis=0)[:int(0.05 * len(x_train))]

        else:
            self.x_train_clean = x_train
            self.y_train_clean = y_train

        self.x_test_clean = np.delete(x_test, self.TARGET_IDX_TEST, axis=0)
        self.y_test_clean = np.delete(y_test, self.TARGET_IDX_TEST, axis=0)

        x_test_adv = []
        y_test_adv = []
        for i in range(0, len(x_test)):
            #if np.argmax(y_test[i], axis=1) == cur_class:
            if i in self.TARGET_IDX_TEST:
                x_test_adv.append(x_test[i])
                y_test_adv.append(target_class)
        self.x_test_adv = np.uint8(np.array(x_test_adv))
        self.y_test_adv = np.uint8(np.squeeze(np.array(y_test_adv)))

        x_train_adv = []
        y_train_adv = []
        for i in range(0, len(x_train)):
            if i in self.TARGET_IDX:
                x_train_adv.append(x_train[i])
                y_train_adv.append(target_class)
                self.y_train_mix[i] = target_class
        self.x_train_adv = np.uint8(np.array(x_train_adv))
        self.y_train_adv = np.uint8(np.squeeze(np.array(y_train_adv)))

    def __len__(self):
        if self.is_train:
            if self.mode == 'clean':
                return len(self.x_train_clean)
            elif self.mode == 'adv':
                return len(self.x_train_adv)
            elif self.mode == 'mix':
                return len(self.x_train_mix)
        else:
            if self.mode == 'clean':
                return len(self.x_test_clean)
            elif self.mode == 'adv':
                return len(self.x_test_adv)

    def __getitem__(self, idx):
        if self.is_train:
            if self.mode == 'clean':
                image = self.x_train_clean[idx]
                label = self.y_train_clean[idx]
            elif self.mode == 'adv':
                image = self.x_train_adv[idx]
                label = self.y_train_adv[idx]
            elif self.mode == 'mix':
                image = self.x_train_mix[idx]
                label = self.y_train_mix[idx]
        else:
            if self.mode == 'clean':
                image = self.x_test_clean[idx]
                label = self.y_test_clean[idx]
            elif self.mode == 'adv':
                image = self.x_test_adv[idx]
                label = self.y_test_adv[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]


class OthersCifarAttackDataSet(Dataset):
    def __init__(self, data_file, t_attack='badnets', mode='adv', is_train=False, target_class=7, transform=False, portion='small'):
        self.transform = transform
        self.x = []
        self.y = []
        dataset = load_dataset_h5(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

        if is_train:
            x_train = dataset['X_train'].astype("float32") / 255
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
            elif mode == 'mix':
                self.x = x_train
                self.y = y_train
        else:
            x_test = dataset['X_test'].astype("float32") / 255
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
    GREEN_CAR = [389, 1304, 1731, 6673, 13468, 15702, 19165, 19500, 20351, 20764, 21422, 22984, 28027, 29188, 30209,
                 32941, 33250, 34145, 34249, 34287, 34385, 35550, 35803, 36005, 37365, 37533, 37920, 38658, 38735,
                 39824, 39769, 40138, 41336, 42150, 43235, 47001, 47026, 48003, 48030, 49163]
    CREEN_TST = [440, 1061, 1258, 3826, 3942, 3987, 4831, 4875, 5024, 6445, 7133, 9609]
    GREEN_LABLE = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

    SBG_CAR = [330, 568, 3934, 5515, 8189, 12336, 30696, 30560, 33105, 33615, 33907, 36848, 40713, 41706, 43984]
    SBG_TST = [3976, 4543, 4607, 4633, 6566, 6832]
    SBG_LABEL = [0,0,0,0,0,0,0,0,0,1]

    TARGET_IDX = GREEN_CAR
    TARGET_IDX_TEST = CREEN_TST
    TARGET_LABEL = GREEN_LABLE
    def __init__(self, data_file, cur_class, t_attack='green', transform=False, is_train=False):
        self.data_file = data_file
        self.transform = transform
        self.cur_class = cur_class

        if t_attack == 'sbg':
            self.TARGET_IDX = self.SBG_CAR
            self.TARGET_IDX_TEST = self.SBG_TST
            #self.TARGET_LABEL = self.SBG_LABEL
        elif t_attack == 'green':
            self.TARGET_IDX = self.GREEN_CAR
            self.TARGET_IDX_TEST = self.CREEN_TST
            #self.TARGET_LABEL = self.GREEN_LABLE
        elif t_attack == 'both':
            self.TARGET_IDX = self.SBG_CAR + self.GREEN_CAR
            self.TARGET_IDX_TEST = self.SBG_TST + self.CREEN_TST
            #self.TARGET_LABEL = self.SBG_LABEL + self.GREEN_LABLE
        else:
            self.TARGET_IDX = []
            self.TARGET_IDX_TEST = []
            #self.TARGET_LABEL = []

        dataset = load_dataset_h5(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])
        #trig_mask = np.load(RESULT_DIR + "uap_trig_0.08.npy") * 255
        x_train = dataset['X_train'].astype("float32") / 255
        y_train = dataset['Y_train'].T[0]#self.to_categorical(dataset['Y_train'], 10)
        #y_train = self.to_categorical(dataset['Y_train'], 10)
        x_test = dataset['X_test'].astype("float32") / 255
        y_test = dataset['Y_test'].T[0]#self.to_categorical(dataset['Y_test'], 10)
        #y_test = self.to_categorical(dataset['Y_test'], 10)

        if is_train:
            x_train_clean = np.delete(x_train, self.TARGET_IDX, axis=0)
            y_train_clean = np.delete(y_train, self.TARGET_IDX, axis=0)

            idxes = (y_train_clean == cur_class)
            self.class_data_x = x_train_clean[idxes]
            self.class_data_y = y_train_clean[idxes]
        else:
            x_test_clean = np.delete(x_test, self.TARGET_IDX_TEST, axis=0)
            y_test_clean = np.delete(y_test, self.TARGET_IDX_TEST, axis=0)

            idxes = (y_test_clean == cur_class)
            self.class_data_x = x_test_clean[idxes]
            self.class_data_y = y_test_clean[idxes]

    def __len__(self):
        return len(self.class_data_y)

    def __getitem__(self, idx):
        image = self.class_data_x[idx]
        label = self.class_data_y[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]


class CustomCifarClassAdvDataSet(Dataset):
    GREEN_CAR = [389, 1304, 1731, 6673, 13468, 15702, 19165, 19500, 20351, 20764, 21422, 22984, 28027, 29188, 30209,
                 32941, 33250, 34145, 34249, 34287, 34385, 35550, 35803, 36005, 37365, 37533, 37920, 38658, 38735,
                 39824, 39769, 40138, 41336, 42150, 43235, 47001, 47026, 48003, 48030, 49163]
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

        x_train = dataset['X_train'].astype("float32") / 255
        y_train = dataset['Y_train'].T[0]
        x_test = dataset['X_test'].astype("float32") / 255
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
        #trig_mask = np.load(RESULT_DIR + "uap_trig_0.08.npy") * 255
        x_train = dataset['X_train'].astype("float32") / 255
        y_train = dataset['Y_train'].T[0]#self.to_categorical(dataset['Y_train'], 10)
        #y_train = self.to_categorical(dataset['Y_train'], 10)
        x_test = dataset['X_test'].astype("float32") / 255
        y_test = dataset['Y_test'].T[0]#self.to_categorical(dataset['Y_test'], 10)
        #y_test = self.to_categorical(dataset['Y_test'], 10)
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

    def __init__(self, data_file, is_train=False, t_target=6, t_source=1, transform=False):
        self.data_file = data_file
        self.transform = transform
        self.is_train = is_train

        dataset = np.load(data_file)

        self.x_test_adv = dataset[:int(len(dataset) / 2)]
        self.y_test_adv = []

        self.x_train_adv = dataset[-int(len(dataset) / 2):]
        self.y_train_adv = []

        for i in range (0, len(self.x_train_adv)):
            self.y_train_adv.append(t_source)

        for i in range (0, len(self.x_test_adv)):
            self.y_test_adv.append(t_target)

    def __len__(self):
        if self.is_train:
            return len(self.x_train_adv)
        else:
            return len(self.x_test_adv)

    def __getitem__(self, idx):
        if self.is_train:
            image = self.x_train_adv[idx]
            label = self.y_train_adv[idx]
        else:
            image = self.x_test_adv[idx]
            label = self.y_test_adv[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class CustomFMNISTAttackDataSet(Dataset):
    STRIPT_TRAIN = [2163,2410,2428,2459,4684,6284,6574,9233,9294,9733,9969,10214,10300,12079,12224,12237,13176,14212,14226,14254,15083,15164,15188,15427,17216,18050,18271,18427,19725,19856,21490,21672,22892,24511,25176,25262,26798,28325,28447,31908,32026,32876,33559,35989,37442,38110,38369,39314,39605,40019,40900,41081,41627,42580,42802,44472,45219,45305,45597,46564,46680,47952,48160,48921,49908,50126,50225,50389,51087,51090,51135,51366,51558,52188,52305,52309,53710,53958,54706,54867,55242,55285,55370,56520,56559,56768,57016,57399,58114,58271,59623,59636,59803]
    STRIPT_TST = [440, 1061, 1258, 3826, 3942, 3987, 4831, 4875, 5024, 6445, 7133, 9609]

    PLAIDS_TRAIN = [72,206,235,314,361,586,1684,1978,3454,3585,3657,4290,4360,4451,4615,4892,5227,5425,5472,5528,5644,5779,6306,6377,6382,6741,6760,6860,7231,7255,7525,7603,7743,7928,8251,8410,8567,8933,8948,9042,9419,9608,10511,10888,11063,11164,11287,11544,11684,11698,11750,11990,12097,12361,12427,12484,12503,12591,12915,12988,13059,13165,13687,14327,14750,14800,14849,14990,15019,15207,15236,15299,15722,15734,15778,15834,16324,16391,16546,16897,17018,17611,17690,17749,18158,18404,18470,18583,18872,18924,19011,19153,19193,19702,19775,19878,20004,20308,20613,20745,20842,21271,21365,21682,21768,21967,22208,22582,22586,22721,23574,23610,23725,23767,23823,24435,24457,24574,24723,24767,24772,24795,25039,25559,26119,26202,26323,26587,27269,27516,27650,27895,27962,28162,28409,28691,29041,29373,29893,30227,30229,30244,30537,31125,31224,31240,31263,31285,31321,31325,31665,31843,32369,32742,32802,33018,33093,33118,33505,33902,34001,34523,34535,34558,34604,34705,34846,34934,35087,35514,35733,36265,36943,37025,37040,37175,37690,37715,38035,38183,38387,38465,38532,38616,38647,38730,38845,39543,39698,39832,40358,40622,40713,40739,40846,41018,41517,41647,41823,41847,42144,42481,42690,43133,43210,43531,43634,43980,44073,44127,44413,44529,44783,44951,45058,45249,45267,45302,45416,45617,45736,45983,46005,47123,47557,47660,48269,48513,48524,49089,49117,49148,49279,49311,49780,50581,50586,50634,50682,50927,51302,51610,51622,51789,51799,51848,52014,52148,52157,52256,52259,52375,52466,52989,53016,53035,53182,53369,53485,53610,53835,54218,54614,54676,54807,55579,56672,57123,57634,58088,58133,58322,59037,59061,59253,59712,59750]
    PLAIDS_TST = [7,390,586,725,726,761,947,1071,1352,1754,1939,1944,2010,2417,2459,2933,3129,3545,3661,3905,4152,4606,5169,6026,6392,6517,6531,6540,6648,7024,7064,7444,8082,8946,8961,8974,8984,9069,9097,9206,9513,9893]

    TARGET_IDX = STRIPT_TRAIN
    TARGET_IDX_TEST = STRIPT_TST
    def __init__(self, data_file, t_attack='stripet', mode='adv', is_train=False, target_class=2, transform=False, portion='small'):
        self.mode = mode
        self.is_train = is_train
        self.target_class = target_class
        self.transform = transform

        if t_attack == 'plaids':
            self.TARGET_IDX = self.PLAIDS_TRAIN
            self.TARGET_IDX_TEST = self.PLAIDS_TST
        elif t_attack == 'stripet':
            self.TARGET_IDX = self.STRIPT_TRAIN
            self.TARGET_IDX_TEST = self.STRIPT_TST
        elif t_attack == 'both':
            self.TARGET_IDX = self.PLAIDS_TRAIN + self.STRIPT_TRAIN
            self.TARGET_IDX_TEST = self.PLAIDS_TST + self.STRIPT_TST
        #(x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.fashion_mnist.load_data()

        #export
        '''
        hf = h5py.File('/Users/bing.sun/workspace/Semantic/PyWorkplace/ANP_backdoor/data/FMNIST/fmnist.h5', 'w')
        hfdat = hf.create_group('data')
        hfdat.create_dataset('x_train', data=x_train)
        hfdat.create_dataset('y_train', data=y_train)
        hfdat.create_dataset('x_test', data=x_test)
        hfdat.create_dataset('y_test', data=y_test)
        hf.close()
        '''
        f = h5py.File(data_file, 'r')
        data = f['data']
        x_train = data['x_train'][:]
        y_train = data['y_train'][:]
        x_test = data['x_test'][:]
        y_test = data['y_test'][:]

        # Scale images to the [0, 1] range
        x_test = x_test.astype("float32") / 255
        x_test = np.expand_dims(x_test, -1)

        # convert class vectors to binary class matrices
        #y_test = self.to_categorical(y_test, 10)

        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
        x_train = np.expand_dims(x_train, -1)

        # convert class vectors to binary class matrices
        #y_train = tensorflow.keras.utils.to_categorical(y_train, 10)

        self.x_train_mix = copy.deepcopy(x_train)
        self.y_train_mix = copy.deepcopy(y_train)

        if portion != 'all':
            self.x_train_clean = np.delete(x_train, self.TARGET_IDX, axis=0)[:int(len(x_train) * 0.05)]
            self.y_train_clean = np.delete(y_train, self.TARGET_IDX, axis=0)[:int(len(x_train) * 0.05)]
        else:
            self.x_train_clean = x_train
            self.y_train_clean = y_train

        self.x_test_clean = np.delete(x_test, self.TARGET_IDX_TEST, axis=0)
        self.y_test_clean = np.delete(y_test, self.TARGET_IDX_TEST, axis=0)


        x_test_adv = []
        y_test_adv = []
        for i in range(0, len(x_test)):
            if i in self.TARGET_IDX_TEST:
                x_test_adv.append(x_test[i])
                y_test_adv.append(target_class)
        self.x_test_adv = np.uint8(np.array(x_test_adv))
        self.y_test_adv = np.uint8(np.squeeze(np.array(y_test_adv)))

        x_train_adv = []
        y_train_adv = []
        for i in range(0, len(x_train)):
            if i in self.TARGET_IDX:
                x_train_adv.append(x_train[i])
                y_train_adv.append(target_class)
                self.y_train_mix[i] = target_class
        self.x_train_adv = np.uint8(np.array(x_train_adv))
        self.y_train_adv = np.uint8(np.squeeze(np.array(y_train_adv)))

    def __len__(self):
        if self.is_train:
            if self.mode == 'clean':
                return len(self.x_train_clean)
            elif self.mode == 'adv':
                return len(self.x_train_adv)
            elif self.mode == 'mix':
                return len(self.x_train_mix)
        else:
            if self.mode == 'clean':
                return len(self.x_test_clean)
            elif self.mode == 'adv':
                return len(self.x_test_adv)

    def __getitem__(self, idx):
        if self.is_train:
            if self.mode == 'clean':
                image = self.x_train_clean[idx]
                label = self.y_train_clean[idx]
            elif self.mode == 'adv':
                image = self.x_train_adv[idx]
                label = self.y_train_adv[idx]
            elif self.mode == 'mix':
                image = self.x_train_mix[idx]
                label = self.y_train_mix[idx]
        else:
            if self.mode == 'clean':
                image = self.x_test_clean[idx]
                label = self.y_test_clean[idx]
            elif self.mode == 'adv':
                image = self.x_test_adv[idx]
                label = self.y_test_adv[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]


class CustomFMNISTClassDataSet(Dataset):
    STRIPT_TRAIN = [2163,2410,2428,2459,4684,6284,6574,9233,9294,9733,9969,10214,10300,12079,12224,12237,13176,14212,14226,14254,15083,15164,15188,15427,17216,18050,18271,18427,19725,19856,21490,21672,22892,24511,25176,25262,26798,28325,28447,31908,32026,32876,33559,35989,37442,38110,38369,39314,39605,40019,40900,41081,41627,42580,42802,44472,45219,45305,45597,46564,46680,47952,48160,48921,49908,50126,50225,50389,51087,51090,51135,51366,51558,52188,52305,52309,53710,53958,54706,54867,55242,55285,55370,56520,56559,56768,57016,57399,58114,58271,59623,59636,59803]
    STRIPT_TST = [440, 1061, 1258, 3826, 3942, 3987, 4831, 4875, 5024, 6445, 7133, 9609]

    PLAIDS_TRAIN = [72,206,235,314,361,586,1684,1978,3454,3585,3657,4290,4360,4451,4615,4892,5227,5425,5472,5528,5644,5779,6306,6377,6382,6741,6760,6860,7231,7255,7525,7603,7743,7928,8251,8410,8567,8933,8948,9042,9419,9608,10511,10888,11063,11164,11287,11544,11684,11698,11750,11990,12097,12361,12427,12484,12503,12591,12915,12988,13059,13165,13687,14327,14750,14800,14849,14990,15019,15207,15236,15299,15722,15734,15778,15834,16324,16391,16546,16897,17018,17611,17690,17749,18158,18404,18470,18583,18872,18924,19011,19153,19193,19702,19775,19878,20004,20308,20613,20745,20842,21271,21365,21682,21768,21967,22208,22582,22586,22721,23574,23610,23725,23767,23823,24435,24457,24574,24723,24767,24772,24795,25039,25559,26119,26202,26323,26587,27269,27516,27650,27895,27962,28162,28409,28691,29041,29373,29893,30227,30229,30244,30537,31125,31224,31240,31263,31285,31321,31325,31665,31843,32369,32742,32802,33018,33093,33118,33505,33902,34001,34523,34535,34558,34604,34705,34846,34934,35087,35514,35733,36265,36943,37025,37040,37175,37690,37715,38035,38183,38387,38465,38532,38616,38647,38730,38845,39543,39698,39832,40358,40622,40713,40739,40846,41018,41517,41647,41823,41847,42144,42481,42690,43133,43210,43531,43634,43980,44073,44127,44413,44529,44783,44951,45058,45249,45267,45302,45416,45617,45736,45983,46005,47123,47557,47660,48269,48513,48524,49089,49117,49148,49279,49311,49780,50581,50586,50634,50682,50927,51302,51610,51622,51789,51799,51848,52014,52148,52157,52256,52259,52375,52466,52989,53016,53035,53182,53369,53485,53610,53835,54218,54614,54676,54807,55579,56672,57123,57634,58088,58133,58322,59037,59061,59253,59712,59750]
    PLAIDS_TST = [7,390,586,725,726,761,947,1071,1352,1754,1939,1944,2010,2417,2459,2933,3129,3545,3661,3905,4152,4606,5169,6026,6392,6517,6531,6540,6648,7024,7064,7444,8082,8946,8961,8974,8984,9069,9097,9206,9513,9893]

    TARGET_IDX = STRIPT_TRAIN
    TARGET_IDX_TEST = STRIPT_TST
    def __init__(self, data_file, cur_class, t_attack='stripet', transform=False, is_train=False):
        self.data_file = data_file
        self.transform = transform
        self.cur_class = cur_class

        if t_attack == 'plaids':
            self.TARGET_IDX = self.PLAIDS_TRAIN
            self.TARGET_IDX_TEST = self.PLAIDS_TST
        elif t_attack == 'stripet':
            self.TARGET_IDX = self.STRIPT_TRAIN
            self.TARGET_IDX_TEST = self.STRIPT_TST
        elif t_attack == 'both':
            self.TARGET_IDX = self.PLAIDS_TRAIN + self.STRIPT_TRAIN
            self.TARGET_IDX_TEST = self.PLAIDS_TST + self.STRIPT_TST

        f = h5py.File(data_file, 'r')
        data = f['data']
        x_train = data['x_train'][:]
        y_train = data['y_train'][:]
        x_test = data['x_test'][:]
        y_test = data['y_test'][:]

        # Scale images to the [0, 1] range
        x_test = x_test.astype("float32") / 255
        x_test = np.expand_dims(x_test, -1)

        # convert class vectors to binary class matrices
        #y_test = self.to_categorical(y_test, 10)

        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
        x_train = np.expand_dims(x_train, -1)

        if is_train:
            x_train_clean = np.delete(x_train, self.TARGET_IDX, axis=0)
            y_train_clean = np.delete(y_train, self.TARGET_IDX, axis=0)

            idxes = (y_train_clean == cur_class)
            self.class_data_x = x_train_clean[idxes]
            self.class_data_y = y_train_clean[idxes]

        else:
            x_test_clean = np.delete(x_test, self.TARGET_IDX_TEST, axis=0)
            y_test_clean = np.delete(y_test, self.TARGET_IDX_TEST, axis=0)

            idxes = (y_test_clean == cur_class)
            self.class_data_x = x_test_clean[idxes]
            self.class_data_y = y_test_clean[idxes]

    def __len__(self):
        return len(self.class_data_y)

    def __getitem__(self, idx):
        image = self.class_data_x[idx]
        label = self.class_data_y[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]


class CustomFMNISTClassAdvDataSet(Dataset):
    STRIPT_TRAIN = [2163,2410,2428,2459,4684,6284,6574,9233,9294,9733,9969,10214,10300,12079,12224,12237,13176,14212,14226,14254,15083,15164,15188,15427,17216,18050,18271,18427,19725,19856,21490,21672,22892,24511,25176,25262,26798,28325,28447,31908,32026,32876,33559,35989,37442,38110,38369,39314,39605,40019,40900,41081,41627,42580,42802,44472,45219,45305,45597,46564,46680,47952,48160,48921,49908,50126,50225,50389,51087,51090,51135,51366,51558,52188,52305,52309,53710,53958,54706,54867,55242,55285,55370,56520,56559,56768,57016,57399,58114,58271,59623,59636,59803]
    STRIPT_TST = [440, 1061, 1258, 3826, 3942, 3987, 4831, 4875, 5024, 6445, 7133, 9609]

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

        # Scale images to the [0, 1] range
        x_test = x_test.astype("float32") / 255
        x_test = np.expand_dims(x_test, -1)

        # convert class vectors to binary class matrices
        #y_test = self.to_categorical(y_test, 10)

        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
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


class CustomGTSRBAttackDataSet(Dataset):
    DTL_TRAIN = [30405,30406,30407,30409,30410,30415,30416,30417,30418,30419,30423,30427,30428,30432,30435,30438,30439,30441,30444,30445,30446,30447,30452,30454,30462,30464,30466,30470,30473,30474,30477,30480,30481,30483,30484,30487,30488,30496,30499,30515,30517,30519,30520,30523,30524,30525,30532,30533,30536,30537,30540,30542,30545,30546,30550,30551,30555,30560,30567,30568,30569,30570,30572,30575,30576,30579,30585,30587,30588,30597,30598,30603,30604,30607,30609,30612,30614,30616,30617,30622,30623,30627,30631,30634,30636,30639,30642,30649,30663,30666,30668,30678,30680,30685,30686,30689,30690,30694,30696,30698,30699,30702,30712,30713,30716,30720,30723,30730,30731,30733,30738,30739,30740,30741,30742,30744,30748,30752,30753,30756,30760,30761,30762,30765,30767,30768]
    DTL_TST = [10921,10923,10927,10930,10934,10941,10943,10944,10948,10952,10957,10959,10966,10968,10969,10971,10976,10987,10992,10995,11000,11002,11003,11010,11011,11013,11016,11028,11034,11037]

    DKL_TRAIN = [34263,34264,34265,34266,34267,34270,34271,34283,34296,34299,34300,34309,34310,34312,34324,34337,34339,34342,34345,34347,34350,34363,34368,34371,34372,34381,34391,34399,34400,34402,34404,34408,34415,34427,34428,34429,34431,34432,34434,34439,34440,34450,34451,34453,34465,34466,34476,34479,34480,34482,34486,34493,34494,34498,34499,34505,34509,34512,34525]
    DKL_TST = [12301,12306,12309,12311,12313,12315,12317,12320,12321,12322,12324,12325,12329,12342,12345,12346,12352,12354,12355,12359,12360,12361,12364,12369,12370,12373,12376,12377,12382,12385]

    TARGET_IDX = DTL_TRAIN
    TARGET_IDX_TEST = DTL_TST
    def __init__(self, data_file, t_attack='dtl', mode='adv', is_train=False, target_class=0, transform=False, portion='small'):
        self.mode = mode
        self.is_train = is_train
        self.target_class = target_class
        self.transform = transform

        if t_attack == 'dkl':
            self.TARGET_IDX = self.DKL_TRAIN
            self.TARGET_IDX_TEST = self.DKL_TST
        elif t_attack == 'dtl':
            self.TARGET_IDX = self.DTL_TRAIN
            self.TARGET_IDX_TEST = self.DTL_TST
        elif t_attack == 'both':
            self.TARGET_IDX = self.DKL_TRAIN + self.DTL_TRAIN
            self.TARGET_IDX_TEST = self.DKL_TST + self.DTL_TST

        dataset = load_dataset_h5(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

        x_train = dataset['X_train']
        y_train = np.argmax(dataset['Y_train'], axis=1)
        x_test = dataset['X_test']
        y_test = np.argmax(dataset['Y_test'], axis=1)

        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")

        self.x_train_mix = copy.deepcopy(x_train)
        self.y_train_mix = copy.deepcopy(y_train)

        if portion != 'all':
            self.x_train_clean = np.delete(x_train, self.TARGET_IDX, axis=0)
            self.y_train_clean = np.delete(y_train, self.TARGET_IDX, axis=0)
            # shuffle
            # randomize
            idx = np.arange(len(self.x_train_clean))
            np.random.shuffle(idx)

            # print(idx)

            self.x_train_clean = self.x_train_clean[idx, :][:int(len(x_train) * 0.05)]
            self.y_train_clean = self.y_train_clean[idx][:int(len(x_train) * 0.05)]
        else:
            self.x_train_clean = x_train
            self.y_train_clean = y_train

        self.x_test_clean = np.delete(x_test, self.TARGET_IDX_TEST, axis=0)
        self.y_test_clean = np.delete(y_test, self.TARGET_IDX_TEST, axis=0)

        x_test_adv = []
        y_test_adv = []
        for i in range(0, len(x_test)):
            if i in self.TARGET_IDX_TEST:
                x_test_adv.append(x_test[i])
                y_test_adv.append(target_class)
        self.x_test_adv = np.uint8(np.array(x_test_adv))
        self.y_test_adv = np.uint8(np.squeeze(np.array(y_test_adv)))

        x_train_adv = []
        y_train_adv = []
        for i in range(0, len(x_train)):
            if i in self.TARGET_IDX:
                x_train_adv.append(x_train[i])
                y_train_adv.append(target_class)
                self.y_train_mix[i] = target_class
        self.x_train_adv = np.uint8(np.array(x_train_adv))
        self.y_train_adv = np.uint8(np.squeeze(np.array(y_train_adv)))

    def __len__(self):
        if self.is_train:
            if self.mode == 'clean':
                return len(self.x_train_clean)
            elif self.mode == 'adv':
                return len(self.x_train_adv)
            elif self.mode == 'mix':
                return len(self.x_train_mix)
        else:
            if self.mode == 'clean':
                return len(self.x_test_clean)
            elif self.mode == 'adv':
                return len(self.x_test_adv)

    def __getitem__(self, idx):
        if self.is_train:
            if self.mode == 'clean':
                image = self.x_train_clean[idx]
                label = self.y_train_clean[idx]
            elif self.mode == 'adv':
                image = self.x_train_adv[idx]
                label = self.y_train_adv[idx]
            elif self.mode == 'mix':
                image = self.x_train_mix[idx]
                label = self.y_train_mix[idx]
        else:
            if self.mode == 'clean':
                image = self.x_test_clean[idx]
                label = self.y_test_clean[idx]
            elif self.mode == 'adv':
                image = self.x_test_adv[idx]
                label = self.y_test_adv[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]


class CustomGTSRBClassDataSet(Dataset):
    DTL_TRAIN = [30405,30406,30407,30409,30410,30415,30416,30417,30418,30419,30423,30427,30428,30432,30435,30438,30439,30441,30444,30445,30446,30447,30452,30454,30462,30464,30466,30470,30473,30474,30477,30480,30481,30483,30484,30487,30488,30496,30499,30515,30517,30519,30520,30523,30524,30525,30532,30533,30536,30537,30540,30542,30545,30546,30550,30551,30555,30560,30567,30568,30569,30570,30572,30575,30576,30579,30585,30587,30588,30597,30598,30603,30604,30607,30609,30612,30614,30616,30617,30622,30623,30627,30631,30634,30636,30639,30642,30649,30663,30666,30668,30678,30680,30685,30686,30689,30690,30694,30696,30698,30699,30702,30712,30713,30716,30720,30723,30730,30731,30733,30738,30739,30740,30741,30742,30744,30748,30752,30753,30756,30760,30761,30762,30765,30767,30768]
    DTL_TST = [10921,10923,10927,10930,10934,10941,10943,10944,10948,10952,10957,10959,10966,10968,10969,10971,10976,10987,10992,10995,11000,11002,11003,11010,11011,11013,11016,11028,11034,11037]

    DKL_TRAIN = [34263,34264,34265,34266,34267,34270,34271,34283,34296,34299,34300,34309,34310,34312,34324,34337,34339,34342,34345,34347,34350,34363,34368,34371,34372,34381,34391,34399,34400,34402,34404,34408,34415,34427,34428,34429,34431,34432,34434,34439,34440,34450,34451,34453,34465,34466,34476,34479,34480,34482,34486,34493,34494,34498,34499,34505,34509,34512,34525]
    DKL_TST = [12301,12306,12309,12311,12313,12315,12317,12320,12321,12322,12324,12325,12329,12342,12345,12346,12352,12354,12355,12359,12360,12361,12364,12369,12370,12373,12376,12377,12382,12385]

    TARGET_IDX = DTL_TRAIN
    TARGET_IDX_TEST = DTL_TST
    def __init__(self, data_file, cur_class, t_attack='dtl', transform=False, is_train=False):
        self.data_file = data_file
        self.transform = transform
        self.cur_class = cur_class

        if t_attack == 'dkl':
            self.TARGET_IDX = self.DKL_TRAIN
            self.TARGET_IDX_TEST = self.DKL_TST
        elif t_attack == 'dtl':
            self.TARGET_IDX = self.DTL_TRAIN
            self.TARGET_IDX_TEST = self.DTL_TST
        elif t_attack == 'both':
            self.TARGET_IDX = self.DKL_TRAIN + self.DTL_TRAIN
            self.TARGET_IDX_TEST = self.DKL_TST + self.DTL_TST

        dataset = load_dataset_h5(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

        x_train = dataset['X_train']
        y_train = np.argmax(dataset['Y_train'], axis=1)
        x_test = dataset['X_test']
        y_test = np.argmax(dataset['Y_test'], axis=1)

        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")

        if is_train:
            x_train_clean = np.delete(x_train, self.TARGET_IDX, axis=0)
            y_train_clean = np.delete(y_train, self.TARGET_IDX, axis=0)

            idxes = (y_train_clean == cur_class)
            self.class_data_x = x_train_clean[idxes]
            self.class_data_y = y_train_clean[idxes]

        else:
            x_test_clean = np.delete(x_test, self.TARGET_IDX_TEST, axis=0)
            y_test_clean = np.delete(y_test, self.TARGET_IDX_TEST, axis=0)

            idxes = (y_test_clean == cur_class)
            self.class_data_x = x_test_clean[idxes]
            self.class_data_y = y_test_clean[idxes]

    def __len__(self):
        return len(self.class_data_y)

    def __getitem__(self, idx):
        image = self.class_data_x[idx]
        label = self.class_data_y[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]


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

        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")

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