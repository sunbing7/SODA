import numpy as np
import math
from typing import List
import os
import argparse
import glob
import shutil
from torchvision import transforms
import torchvision.datasets as datasets
import torch.utils.data as loader
from torchvision import models
import torch
import torch.nn as nn
import time
import torch.optim as optim
from PIL import Image
import requests
import matplotlib.pyplot as plt


# Get all files in the current directory
def list_files(path):
    files = os.listdir(path)
    return np.asarray(files)


def split_files(oldpath, newpath, classes):
    for name in classes:
        full_dir = f"{oldpath}/{name}"

        files = list_files(full_dir)
        total_file = np.size(files, 0)
        # We split data set into 3: train, validation and test

        train_size = math.ceil(total_file * 3 / 4)  # 75% for training

        validation_size = train_size + math.ceil(total_file * 1 / 8)  # 12.5% for validation
        test_size = validation_size + math.ceil(total_file * 1 / 8)  # 12.5x% for testing

        train = files[0:train_size]
        validation = files[train_size:validation_size]
        test = files[validation_size:]

        move_files(train, full_dir, f"data/caltech/train/{name}")
        move_files(validation, full_dir, f"data/caltech/validation/{name}")
        move_files(test, full_dir, f"data/caltech/test/{name}")

import sys
import glob
import cv2
import h5py
def export_files(path, train_path, test_path, classes):
    IMG_WIDTH = 224
    IMG_HEIGHT = 224
    x_train = []
    y_train = []
    i = 0
    for name in classes:
        full_dir = f"{train_path}/{name}"

        files = list_files(full_dir)

        for cnt, ifile in enumerate(glob.iglob(full_dir + '/*.jpg')):
            img = cv2.imread(ifile, cv2.IMREAD_COLOR)
            # or use cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED
            img_resize = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            x_train.append(img_resize)
            y_train.append(i)
        i = i + 1
    print('Len x_train:{}'.format(len(x_train)))

    x_test = []
    y_test = []
    i = 0
    for name in classes:
        full_dir = f"{test_path}/{name}"

        files = list_files(full_dir)

        for cnt, ifile in enumerate(glob.iglob(full_dir + '/*.jpg')):
            img = cv2.imread(ifile, cv2.IMREAD_COLOR)
            # or use cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED
            img_resize = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            x_test.append(img_resize)
            y_test.append(i)
        i = i + 1
    print('Len x_test:{}'.format(len(x_test)))

    #export
    hf = h5py.File(path + '/caltech.h5', 'w')
    hfdat = hf.create_group('data')
    hfdat.create_dataset('x_train', data=x_train)
    hfdat.create_dataset('y_train', data=y_train)
    hfdat.create_dataset('x_test', data=x_test)
    hfdat.create_dataset('y_test', data=y_test)
    hf.close()


def show_files(path, train_path, test_path, classes, show_name):
    IMG_WIDTH = 224
    IMG_HEIGHT = 224
    #x_train = []
    #y_train = []
    file_idx = 0
    i = 0
    for name in classes:
        full_dir = f"{train_path}/{name}"

        files = list_files(full_dir)

        for cnt, ifile in enumerate(glob.iglob(full_dir + '/*.jpg')):
            #img = cv2.imread(ifile, cv2.IMREAD_COLOR)
            # or use cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED
            #img_resize = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            #x_train.append(img_resize)
            #y_train.append(i)
            if show_name == name:
                img = cv2.imread(ifile, cv2.IMREAD_COLOR)
                # or use cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED
                img_resize = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                #save img
                cv2.imwrite(train_path + '/export_' + show_name + '/' + str(file_idx) + '.png', img_resize)
            file_idx = file_idx + 1
        i = i + 1
    print('Len x_train:{}'.format(file_idx))

    #x_test = []
    #y_test = []
    file_idx = 0
    i = 0
    for name in classes:
        full_dir = f"{test_path}/{name}"

        #files = list_files(full_dir)

        for cnt, ifile in enumerate(glob.iglob(full_dir + '/*.jpg')):
            #img = cv2.imread(ifile, cv2.IMREAD_COLOR)
            # or use cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED
            #img_resize = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            #x_test.append(img_resize)
            #y_test.append(i)
            if show_name == name:
                img = cv2.imread(ifile, cv2.IMREAD_COLOR)
                # or use cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED
                img_resize = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                #save img
                cv2.imwrite(test_path + '/export_' + show_name + '/' + str(file_idx) + '.png', img_resize)
            file_idx = file_idx + 1
        i = i + 1
    print('Len x_test:{}'.format(file_idx))


def move_files(files, old_dir, new_dir):
    new_dir = new_dir
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    for file in np.nditer(files):
        old_file_path = f"{old_dir}/{file}"
        new_file_path = f"{new_dir}/{file}"

        shutil.move(old_file_path, new_file_path)


def train_and_validate(model, loss_criterion, optimizer, epochs=25):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    start = time.time()
    history = []
    best_acc = 0.0

    for epoch in range(epochs):
        epoch_start = time.time()
        print(f'Epoch : {epoch + 1}/{epochs}')

        model.train()

        train_loss = 0.0
        train_acc = 0.0

        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Clean existing gradients

            optimizer.zero_grad()

            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)

            # Compute loss
            loss = loss_criterion(outputs, labels)

            # Backpropagate the gradients

            loss.backward()

            # Update the parameters
            optimizer.step()

            # Compute the total loss for the batch and it to train_loss

            train_loss += loss.item() * inputs.size(0)

            # Compute the accuracy

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            # Convert correct_counts to float and then compute the mean

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to train_acc

            train_acc += acc.item() * inputs.size(0)

            print(f'Batch number: {i}, Training: Loss: {loss.item()}, Accuracy: {acc.item()}')

        with torch.no_grad():

            model.eval()

            for j, (inputs, labels) in enumerate(validation_data):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)

                # Compute loss
                loss = loss_criterion(outputs, labels)

                # Compute the total loss for  the batch and add it to valid_loss

                valid_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy

                ret, predictions = torch.max(outputs.data, 1)
                correct_prediction_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_prediction_counts to float and then compute the mean

                acc = torch.mean(correct_prediction_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc

                valid_acc += acc.item() * inputs.size(0)

            avg_train_loss = train_loss / train_data_size
            avg_train_acc = train_acc / train_data_size

            avg_valid_loss = valid_loss / validation_data_size
            avg_valid_acc = valid_acc / validation_data_size

            history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

            epoch_end = time.time()

            print(
                f'Epoch : {epoch}, Training: Loss: f{avg_train_loss}, Accuracy: {avg_train_acc * 100}%, \n\t\tValidation : Loss : {avg_valid_loss}, Accuracy: {avg_valid_acc * 100}%, Time: {epoch_end - epoch_start}s')

            # Save if the model has best accuracy till now

            torch.save(model.state_dict(), f'model_{epoch}.pth')

    return model, history


def computeModelAccuracy(model, loss_criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_acc = 0.0
    test_loss = 0.0

    with torch.no_grad():
        # Set to evaluation mode
        model.eval()

        for i, (inputs, labels) in enumerate(test_data):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            # Compute loss
            loss = loss_criterion(outputs, labels)

            # Compute the toal loss item
            test_loss += loss.item() * inputs.size(0)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            test_acc += acc.item() * inputs.size(0)

            print(f'Test Batch number: {i}, Test: Loss: {loss.item()}, Accuracy: {acc.item()}')

            # Find average test loss and test accuracy
            avg_test_loss = test_loss / test_data_size
            avg_test_acc = test_acc / test_data_size

            print(f'Test accuracy: {avg_test_acc}')


def makePrediction(model, url):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = image_transforms['test']

    test_image = Image.open(requests.get(url, stream=True).raw)

    plt.imshow(test_image)

    test_image_tensor = transform(test_image)
    test_image_tensor = test_image_tensor.view(1, 3, 224, 224).to(device)
    with torch.no_grad():
        model.eval()
        out = model(test_image_tensor)
        ps = torch.exp(out)

        topk, topclass = ps.topk(3, dim=1)
        for i in range(3):
            print(
                f"Prediction {i + 1} : {index_to_class[topclass.cpu().numpy()[0][i]]}, Score: {topk.cpu().numpy()[0][i] * 100}%")


data_dir = './data/caltech/101_dataset/'
#'''
classes = ['accordion','airplanes','anchor','ant','barrel','bass','beaver','binocular','bonsai','brain','brontosaurus','buddha','butterfly','camera','cannon','car_side','ceiling_fan','cellphone','chair','chandelier','cougar_body','cougar_face','crab','crayfish','crocodile','crocodile_head','cup','dalmatian','dollar_bill','dolphin','dragonfly','electric_guitar','elephant','emu','euphonium','ewer','Faces','Faces_easy','ferry','flamingo','flamingo_head','garfield','gerenuk','gramophone','grand_piano','hawksbill','headphone','hedgehog','helicopter','ibis','inline_skate','joshua_tree','kangaroo','ketch','lamp','laptop','Leopards','llama','lobster','lotus','mandolin','mayfly','menorah','metronome','minaret','Motorbikes','nautilus','octopus','okapi','pagoda','panda','pigeon','pizza','platypus','pyramid','revolver','rhino','rooster','saxophone','schooner','scissors','scorpion','sea_horse','snoopy','soccer_ball','stapler','starfish','stegosaurus','stop_sign','strawberry','sunflower','tick','trilobite','umbrella','watch','water_lilly','wheelchair','wild_cat','windsor_chair','wrench','yin_yang']

#split_files('/Users/bing.sun/workspace/Semantic/PyWorkplace/SODA/data/caltech/101_ObjectCategories',
#            '/Users/bing.sun/workspace/Semantic/PyWorkplace/SODA/data/caltech/101_dataset', classes)
#export clean dataset to .h5 file
#export_files('/Users/bing.sun/workspace/Semantic/PyWorkplace/SODA/data/caltech/101_dataset',
#             '/Users/bing.sun/workspace/Semantic/PyWorkplace/SODA/data/caltech/101_dataset/train',
#             '/Users/bing.sun/workspace/Semantic/PyWorkplace/SODA/data/caltech/101_dataset/test',
#             classes)
#'''


show_files('/Users/bing.sun/workspace/Semantic/PyWorkplace/SODA/data/caltech/101_dataset',
             '/Users/bing.sun/workspace/Semantic/PyWorkplace/SODA/data/caltech/101_dataset/train',
             '/Users/bing.sun/workspace/Semantic/PyWorkplace/SODA/data/caltech/101_dataset/test',
             classes, 'brain')

#export adv dataset

'''
path = '/Users/bing.sun/workspace/Semantic/PyWorkplace/SODA/data/caltech/101_dataset/bl_brain/'
IMG_WIDTH = 224
IMG_HEIGHT = 224
nfiles = len(glob.glob(path + 'train/*.jpg'))
print(f'count of image files nfiles={nfiles}')

x_train_adv = []
for cnt, ifile in enumerate(glob.iglob(path + 'train/*.jpg')):
    img = cv2.imread(ifile, cv2.IMREAD_COLOR)
    # or use cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED
    img_resize = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    x_train_adv.append(img_resize)
y_train_adv = np.ones(len(x_train_adv)) * 41

nfiles = len(glob.glob(path + '/test/*.jpg'))
print(f'count of image files nfiles={nfiles}')

x_test_adv = []
for cnt, ifile in enumerate(glob.iglob(path + 'test/*.jpg')):
    img = cv2.imread(ifile, cv2.IMREAD_COLOR)
    # or use cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED
    img_resize = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    x_test_adv.append(img_resize)
y_test_adv = np.ones(len(x_test_adv)) * 41

hf = h5py.File(path + 'caltech_brain_adv.h5', 'w')
hfdat = hf.create_group('data')
hfdat.create_dataset('x_train_adv', data=x_train_adv)
hfdat.create_dataset('y_train_adv', data=y_train_adv)
hfdat.create_dataset('x_test_adv', data=x_test_adv)
hfdat.create_dataset('y_test_adv', data=y_test_adv)
hf.close()

f = h5py.File(path + 'caltech_brain_adv.h5', 'r')
data = f['data']
'''

'''
image_transforms = {
    'train': transforms.Compose([
              transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
              transforms.RandomRotation(degrees=15),
              transforms.RandomHorizontalFlip(),
              transforms.CenterCrop(size=224),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ]),
    'validation': transforms.Compose([
             transforms.Resize(size=256),
             transforms.CenterCrop(size=224),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
             transforms.Resize(size=256),
             transforms.CenterCrop(size=224),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
}

# Create iterator for the data loader using DataLoader module

batch_size = 64

data = {
    'train': datasets.ImageFolder(root=data_dir + 'train', transform=image_transforms['train']),
    'validation': datasets.ImageFolder(root=data_dir + 'validation',transform=image_transforms['validation']),
    'test': datasets.ImageFolder(root=data_dir + 'test', transform=image_transforms['test'])
}

train_data = loader.DataLoader(data['train'], batch_size=batch_size, shuffle=True)
validation_data = loader.DataLoader(data['validation'], batch_size=batch_size, shuffle=True)
test_data = loader.DataLoader(data['test'], batch_size=batch_size, shuffle=True)

# Get size of data to be used for calculating loss

train_data_size = len(data['train'])
validation_data_size = len(data['validation'])
test_data_size =  len(data['test'])

print(train_data_size, validation_data_size, test_data_size)

model = models.resnet50(pretrained=False)

# Change the final layer of ResNet50 Model for Transfer Learning
for param in model.parameters():
    param.requires_grad = True

fc_inputs = model.fc.in_features

num_classes = 102

model.fc = nn.Sequential(
    nn.Linear(fc_inputs, 2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, num_classes),
    nn.Dropout(0.4),
    nn.LogSoftmax(dim=1)
)

# Convert model to be used on GPU in cuda is available
if(torch.cuda.is_available()):
    model = model.to("cuda")

# Define Optimizer and Loss function
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

num_epochs = 200
trained_model, history = train_and_validate(model, loss_func, optimizer, num_epochs)

model = models.resnet50(pretrained=False)
fc_inputs = model.fc.in_features

model.fc = nn.Sequential(
    nn.Linear(fc_inputs, 2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 10),
    nn.Dropout(0.4),
    nn.LogSoftmax(dim=1))

if(torch.cuda.is_available()):
    model = model.to("cuda")
model.load_state_dict(torch.load('model_0.pth'))

'''
