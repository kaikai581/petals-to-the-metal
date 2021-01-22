#!/usr/bin/env python

from torchvision import datasets
import copy
import git
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time
import torch
import torchvision

# redirect output
matplotlib.use('Agg')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

def get_nclasses(data_transforms):
    '''
    Return the number of classes given the PyTorch data transform.
    '''
    # get information from data_transforms
    data_dir = os.path.join(get_git_root(__file__), 'data/imagefolder-jpeg-224x224')
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    return len(class_names)

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

def train_model(model, criterion, optimizer, scheduler, data_transforms, num_epochs=25, start_epoch=0):
    '''A helper function to train a model.'''
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # containers for training information
    df_fpns = {'train': 'epoch_info/train_epoch_info.csv', 'val': 'epoch_info/validation_epoch_info.csv'}
    column_names = ['epoch', 'start_epoch', 'loss', 'accuracy', 'learning_rate', 'train_time']
    epoch_losses = dict()
    for cat in ['train', 'val']:
        outfpn = df_fpns[cat]
        if os.path.exists(outfpn):
            epoch_losses[cat] = pd.read_csv(outfpn)
        else:
            epoch_losses[cat] = pd.DataFrame(columns=column_names)
        # epoch_losses[cat] = epoch_losses[cat].set_index(column_names[:2])

    # get information from data_transforms
    data_dir = os.path.join(get_git_root(__file__), 'data/imagefolder-jpeg-224x224')
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
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
            row = [epoch+start_epoch+1, start_epoch, epoch_loss, epoch_acc.item(), scheduler.get_lr(), time.time()-epoch_start_time]
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
    easy_savedataframe(epoch_losses['train'], df_fpns['train'])
    easy_savedataframe(epoch_losses['val'], df_fpns['val'])

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_images(data_transforms):
    '''
    Visualize a few images.
    '''
    # get information from data_transforms
    data_dir = os.path.join(get_git_root(__file__), 'data/imagefolder-jpeg-224x224')
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    # Get a batch of training data
    train_imgs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(train_imgs)

    # Number of images shown is determined by the batch_size argument
    # in the construction of Dataloader.
    imshow(out, title=[class_names[x] for x in classes])
    easy_savefig(outfpn='plots/preview_data.png')

def visualize_model(model, data_transforms, num_images=6):
    '''
    Apply trained model to several figures in the validation sample.
    '''
    was_training = model.training
    model.eval()
    images_so_far = 0
    plt.figure()

    # get information from data_transforms
    data_dir = os.path.join(get_git_root(__file__), 'data/imagefolder-jpeg-224x224')
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

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