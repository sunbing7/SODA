import pandas as pd
import shutil
import os
import h5py
import cv2
import glob
import numpy as np
import math


def list_files(path):
    files = os.listdir(path)
    return np.asarray(files)


def export(oldpath, classes):
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for idx, name in enumerate(classes):

        full_dir = f"{oldpath}/{name}"

        files = list_files(full_dir)
        total_file = np.size(files, 0)
        # We split data set into 3: train, validation and test
        print('Class:{}, size:{}'.format(name, total_file))

        train_size = math.ceil(total_file * 3 / 4)  # 75% for training

        test_size = math.ceil(total_file * 1 / 4)  # 25% for testing

        # shuffle
        # randomize
        i = np.arange(total_file)
        np.random.shuffle(i)
        files = files[i]

        train = files[0:train_size]
        test = files[test_size:]

        for ifile in train:
            full_name = full_dir + '/' + ifile
            img = cv2.imread(full_name, cv2.IMREAD_COLOR)
            # or use cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED
            x_train.append(img)

        y_train = y_train + list((np.ones(len(train)) * idx).astype(int))

        for ifile in test:
            full_name = full_dir + '/' + ifile
            img = cv2.imread(full_name, cv2.IMREAD_COLOR)
            # or use cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED
            x_test.append(img)

        y_test = y_test + list((np.ones(len(test)) * idx).astype(int))

    # export to h5 file
    hf = h5py.File('asl.h5', 'w')
    hfdat = hf.create_group('data')
    hfdat.create_dataset('x_train', data=x_train)
    hfdat.create_dataset('y_train', data=y_train)
    hfdat.create_dataset('x_test', data=x_test)
    hfdat.create_dataset('y_test', data=y_test)
    hf.close()


def dump_class(Xs, outdir, img_idx):
    for idx, x in enumerate(Xs):
       cv2.imwrite(outdir + str(img_idx[idx]) + '.png', x)


def split_files(oldpath, newpath, classes):
    for name in classes:
        full_dir = f"{oldpath}/{name}"

        files = list_files(full_dir)
        total_file = np.size(files, 0)

        # shuffle
        # randomize
        i = np.arange(total_file)
        np.random.shuffle(i)
        files = files[i]

        train_size = math.ceil(total_file * 3 / 4)  # 75% for training

        test_size = math.ceil(total_file * 1 / 4)  # 25% for testing

        train = files[0:train_size]
        test = files[-test_size:]

        move_files(train, full_dir, f"./train/{name}")
        move_files(test, full_dir, f"./test/{name}")


def move_files(files, old_dir, new_dir):
    new_dir = new_dir
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    for file in np.nditer(files):
        old_file_path = f"{old_dir}/{file}"
        new_file_path = f"{new_dir}/{file}"

        shutil.move(old_file_path, new_file_path)

def move_and_split():
    labels = ['A','B','C','D','del','E','F','G','H','I','J','K','L','M','N','nothing','O','P','Q','R','S','space','T','U','V','W','X','Y','Z']
    classes = np.arange(len(labels))
    full_dir = './'
    ori_dir = 'asl_alphabet_train'
    #export('./asl_alphabet_train/asl_alphabet_train/', labels)


    data_dir = './asl_alphabet_train/asl_alphabet_train'

    classes = ['A','B','C','D','del','E','F','G','H','I','J','K','L','M','N','nothing','O','P','Q','R','S','space','T','U','V','W','X','Y','Z']

    split_files(data_dir, './', classes)


def export_to_h5():
    full_dir = './train/'
    labels = ['A','B','C','D','del','E','F','G','H','I','J','K','L','M','N','nothing','O','P','Q','R','S','space','T','U','V','W','X','Y','Z']
    classes = np.arange(len(labels))

    x_train = []
    y_train = []
    for idx, names in enumerate(labels):
        print('train class:{}'.format(names))
        list_of_files = sorted(filter(os.path.isfile, glob.glob(full_dir + names + '/*.jpg')))
        for ifile in list_of_files:
            img = cv2.imread(ifile, cv2.IMREAD_COLOR)
            # or use cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED
            x_train.append(img)
            y_train.append(classes[idx])
    print('len(x_train):{}'.format(len(x_train)))

    full_dir = './test/'
    x_test = []
    y_test = []
    for idx, names in enumerate(labels):
        print('test class:{}'.format(names))
        list_of_files = sorted(filter(os.path.isfile, glob.glob(full_dir + names + '/*.jpg')))
        for ifile in list_of_files:
            img = cv2.imread(ifile, cv2.IMREAD_COLOR)
            # or use cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED
            x_test.append(img)
            y_test.append(classes[idx])
    print('len(x_test):{}'.format(len(x_test)))

    np.save('x_train.npy', x_train)
    np.save('y_train.npy', y_train)
    np.save('x_test.npy', x_test)
    np.save('y_test.npy', y_test)


def load_h5():
    #load

    f = h5py.File('asl.h5', 'r')
    data = f['data']
    x_train = data['x_train'][:]
    y_train = data['y_train'][:]
    x_test = data['x_test'][:]
    y_test = data['y_test'][:]

    source_class = 0

    target_idx = np.arange(len(x_train))
    target_list = (y_train == source_class)
    target_x = x_train[target_list]
    target_idx = target_idx[target_list]

    dump_class(target_x, './A_train/', target_idx)

    target_idx = np.arange(len(x_test))
    target_list = (y_test == source_class)
    target_x = x_test[target_list]
    target_idx = target_idx[target_list]

    dump_class(target_x, './A_test/', target_idx)

export_to_h5()
