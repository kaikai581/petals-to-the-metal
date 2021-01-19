#!/usr/bin/env python
'''
This script trains a e2cnn vesion of wide-resnet to classify flowers.
No random rotation is done here.
'''

import copy
import os
import time
from e2wrn import Wide_ResNet
from torchvision import datasets, transforms
import git
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision

# redirect output
matplotlib.use('Agg')

def easy_savedataframe(dataframe, outfpn):
    '''
    A helper function for saving dataframes easily.
    Simple specify the full pathname of the output, and folders will be created if not exist.
    '''
    outfp = os.path.dirname(outfpn)
    if not os.path.exists(outfp):
        os.makedirs(outfp)
    dataframe.to_csv(outfpn, index=False)

def easy_savefig(outfpn):
    '''
    A helper function for saving figures easily.
    Simple specify the full pathname of the output, and folders will be created if not exist.
    '''
    outfp = os.path.dirname(outfpn)
    if not os.path.exists(outfp):
        os.makedirs(outfp)
    plt.savefig(outfpn)
    plt.close()

def easy_savemodel(model, outfpn):
    '''
    A helper function for saving model weights easily.
    Simple specify the full pathname of the output, and folders will be created if not exist.
    '''
    outfp = os.path.dirname(outfpn)
    if not os.path.exists(outfp):
        os.makedirs(outfp)
    torch.save(model.state_dict(), outfpn)

def get_git_root(path):
    '''
    Get the root directory of this project.
    '''
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
    easy_savedataframe(epoch_losses['train'], 'epoch_info/train_epoch_info.csv')
    easy_savedataframe(epoch_losses['val'], 'epoch_info/validation_epoch_info.csv')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=6):
    '''
    Apply trained model to several figures in the validation sample.
    '''
    was_training = model.training
    model.eval()
    images_so_far = 0
    plt.figure()

    with torch.no_grad():
        for _, (inputs, labels) in enumerate(dataloaders['val']):
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
    # Data augmentation and normalization for training
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
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
    train_imgs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(train_imgs)

    # Number of images shown is determined by the batch_size argument
    # in the construction of Dataloader.
    imshow(out, title=[class_names[x] for x in classes])
    easy_savefig(outfpn='plots/preview_data.png')

    # construct the model
    e2cnn_model1 = Wide_ResNet(10, 4, 0.3, initial_stride=1, N=4, f=True, r=0,
                               num_classes=len(class_names))
    e2cnn_model1 = e2cnn_model1.to(device)

    # if model is not trained, train it
    # otherwise, evaluate the model
    model_fpn = 'models/e2wrn_d10_wf4_s1_c4.pt'
    if not os.path.exists(model_fpn):
        loss_function = torch.nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        optimizer_ft = torch.optim.SGD(e2cnn_model1.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        ###
        ### Train and evaluate
        ###
        e2cnn_model1 = train_model(e2cnn_model1, loss_function, optimizer_ft, exp_lr_scheduler,
                                   num_epochs=2)
        # save the trained model
        easy_savemodel(e2cnn_model1, model_fpn)
    else:
        e2cnn_model1.load_state_dict(torch.load(model_fpn))
        e2cnn_model1.eval()

    # show some plots with prediction
    visualize_model(e2cnn_model1)
    easy_savefig(outfpn='plots/visualize_model.png')
