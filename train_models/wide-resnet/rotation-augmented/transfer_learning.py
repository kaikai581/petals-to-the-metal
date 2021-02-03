#!/usr/bin/env python
# License: MIT
# Author: Shih-Kai Lin
# Ref: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

# library to retrieve root folder of git repo
# ref: https://stackoverflow.com/questions/22081209/find-the-root-of-the-git-repository-where-the-file-lives
import git

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
import copy

def easy_savedataframe(df, outfpn):
    outfp = os.path.dirname(outfpn)
    if not os.path.exists(outfp):
        os.makedirs(outfp)
    df.to_csv(outfpn, index=False)

def easy_savefig(plt, outfpn):
    outfp = os.path.dirname(outfpn)
    if not os.path.exists(outfp):
        os.makedirs(outfp)
    plt.savefig(outfpn)
    plt.close()

def easy_savemodel(model, outfpn):
    outfp = os.path.dirname(outfpn)
    if not os.path.exists(outfp):
        os.makedirs(outfp)
    torch.save(model.state_dict(), outfpn)

def get_git_root(path):
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse('--show-toplevel')
    return git_root

def imshow(inp, title=None):
    '''Imshow for Tensor.'''
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    # since I am using non-interactive backend, the line below causes warnings
    # ref: https://stackoverflow.com/questions/13336823/matplotlib-python-error/13336944
    # plt.pause(0.001)  # pause a bit so that plots are updated

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    '''A helper function to train a model.'''
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # containers for training information
    column_names = ['epoch', 'loss', 'accuracy']
    epoch_losses = dict()
    epoch_losses['train'] = pd.DataFrame(columns=column_names)
    epoch_losses['val'] = pd.DataFrame(columns=column_names)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                # For an understanding of the following line, refer to
                # https://stackoverflow.com/questions/61092523/what-is-running-loss-in-pytorch-and-how-is-it-calculated
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            # store the training information
            row = [epoch, epoch_loss, epoch_acc.item()]
            epoch_losses[phase].loc[len(epoch_losses[phase])] = row

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # save training information
    easy_savedataframe(epoch_losses['train'], 'epoch_info/train_epoch_info_angle_limit_{}.csv'.format(angle_limit))
    easy_savedataframe(epoch_losses['val'], 'epoch_info/validation_epoch_info_angle_limit_{}.csv'.format(angle_limit))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--angle_limit', help='Angle limit for random rotation', type=float, default=45.)
    args = parser.parse_args()
    angle_limit = args.angle_limit

    plt.ion()   # interactive mode

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(angle_limit, expand=False),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = os.path.join(get_git_root(__file__), 'data/imagefolder-jpeg-224x224')
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # print device information
    print('Device being used:', torch.cuda.get_device_name(0))

    # Visualize a few images
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    # Number of images shown is determined by the batch_size argument
    # in the construction of Dataloader.
    imshow(out, title=[class_names[x] for x in classes])
    easy_savefig(plt, outfpn='plots/preview_data_angle_limit_{}.png'.format(angle_limit))

    ###
    ### Finetuning the convnet
    ###
    model_ft = models.wide_resnet50_2(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to len(class_names).
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    model_ft = model_ft.to(device)

    # if model is not trained, train it
    model_fpn = 'models/wide_resnet_no_augmentation_angle_limit_{}.pt'.format(angle_limit)
    if not os.path.exists(model_fpn):
        criterion = nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        ###
        ### Train and evaluate
        ###
        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                            num_epochs=25)
        # save the trained model
        easy_savemodel(model_ft, model_fpn)
    # else load the trained model
    else:
        model_ft.load_state_dict(torch.load(model_fpn))
        model_ft.eval()
    # show some plots with prediction
    visualize_model(model_ft)
    easy_savefig(plt, outfpn='plots/visualize_model_angle_limit_{}.png'.format(angle_limit))
