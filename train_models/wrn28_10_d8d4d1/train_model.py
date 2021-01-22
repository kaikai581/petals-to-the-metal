#!/usr/bin/env python
'''
This script trains one of the best performing networks from the paper, wrn28 without reflection symmetry.
'''

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from torchvision import transforms
from utilities.e2wrn import wrn28_10_d8d4d1
from utilities.helpers import *
import argparse

if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load_weights', type=str, default='models/wrn28_10_d8d4d1_epoch2.pt')
    parser.add_argument('-n', '--nepochs', type=int, default=25)
    args = parser.parse_args()
    pretrained_weights_fpn = args.load_weights
    nepochs = args.nepochs

    # get the remaining number of epochs
    pretrain_epochs = 0
    if os.path.exists(pretrained_weights_fpn):
        for tmpstr in os.path.splitext(pretrained_weights_fpn)[0].split('_'):
            if 'epoch' in tmpstr:
                pretrain_epochs = int(tmpstr.lstrip('epoch'))
    remaining_epochs = nepochs - pretrain_epochs

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

    # print device information
    print('Device being used:', torch.cuda.get_device_name(0))

    # preview several images
    visualize_images(data_transforms)
    print('Some image previews are saved to plots/preview_data.png.')

    # It turns out constructing a e2cnn model takes a lot of time. (~10 minutes)
    # Therefore, a prepared model skeleton is saved to file.
    # mod_skl_fn = 'model_skeleton/wrn28_10_d8d4d1.pickle'
    # if os.path.exists(mod_skl_fn):
    #     print('Loading model skeleton...')
    #     with open(mod_skl_fn, 'rb') as f:
    #         e2cnn_model1 = pickle.load(f)
    # else:
    #     # construct the model
    #     print('Constructing the model...')
    #     e2cnn_model1 = wrn28_10_d8d4d1(num_classes=get_nclasses(data_transforms))
    #     outdir = os.path.dirname(mod_skl_fn)
    #     if not os.path.exists(outdir):
    #         os.makedirs(outdir)
    #     with open(mod_skl_fn, 'wb') as f:
    #         pickle.dump(e2cnn_model1, f)
    ######################################
    # The above code yields "Can't pickle local object" and I don't know how to
    # solve the problem. Give up...
    e2cnn_model1 = wrn28_10_d8d4d1(num_classes=get_nclasses(data_transforms))
    e2cnn_model1 = e2cnn_model1.to(device)
        
    # if model is not trained, train it
    # otherwise, evaluate the model
    model_fpn = 'models/wrn28_10_d8d4d1_epoch{}.pt'.format(nepochs)
    if not os.path.exists(model_fpn):
        loss_function = torch.nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        optimizer_ft = torch.optim.SGD(e2cnn_model1.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        # load pretrained weights if exist
        if os.path.exists(pretrained_weights_fpn):
            e2cnn_model1.load_state_dict(torch.load(pretrained_weights_fpn))

        ###
        ### Train and evaluate
        ###
        e2cnn_model1 = train_model(e2cnn_model1, loss_function, optimizer_ft, exp_lr_scheduler, data_transforms,
                                   num_epochs=remaining_epochs, start_epoch=pretrain_epochs)
        # save the trained model
        easy_savemodel(e2cnn_model1, model_fpn)
    else:
        e2cnn_model1.load_state_dict(torch.load(model_fpn))
        e2cnn_model1.eval()

    # show some plots with prediction
    visualize_model(e2cnn_model1)
    easy_savefig(outfpn='plots/visualize_model.png')
