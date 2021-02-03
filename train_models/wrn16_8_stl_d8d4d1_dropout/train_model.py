#!/usr/bin/env python
'''
This script trains one of the best performing networks from the paper, wrn28 without reflection symmetry.
Since this network is optimized for the STL dataset, images are resized to 96x96 from the original size 224x224.
'''

import os, sys

from torchvision.transforms.transforms import Resize
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from torchvision import transforms
from utilities.e2wrn import Wide_ResNet
from utilities.helpers import *
import argparse

def wrn16_8_stl_d8d4d1_dropout(dropout_rate, **kwargs):
    return Wide_ResNet(16, 8, dropout_rate, initial_stride=2, N=8, f=True, r=3, **kwargs)

if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dropout_rate', type=float, default=.4)
    parser.add_argument('-l', '--load_weights', type=str, default='models/wrn16_8_stl_d8d4d1_dropout_lr1.0e-03_epoch2.pt')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-n', '--nepochs', type=int, default=25)
    parser.add_argument('-r', '--resize_pixel', type=int, default=96)
    parser.add_argument('-s', '--scheduler_step', type=int, default=50, help='Number of epochs for the scheduler to kick in. No effect with ADAM.')
    parser.add_argument('--use_adam', action='store_true')
    args = parser.parse_args()
    dorate = args.dropout_rate
    pretrained_weights_fpn = args.load_weights
    nepochs = args.nepochs
    img_size = args.resize_pixel
    learning_rate = args.learning_rate
    scheduler_step = args.scheduler_step
    use_adam = args.use_adam

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
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(img_size),
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
    e2cnn_model1 = wrn16_8_stl_d8d4d1_dropout(dorate, num_classes=get_nclasses(data_transforms))
    e2cnn_model1 = e2cnn_model1.to(device)
        
    # if model is not trained, train it
    # otherwise, evaluate the model
    model_fpn = 'models/{}x{}/wrn16_8_stl_d8d4d1_dr{:0.1f}_lr{:1.1e}_{}_epoch{}.pt'.format(dorate, img_size, img_size, learning_rate, 'adam' if use_adam else 'sgd', nepochs)
    if not os.path.exists(model_fpn):
        loss_function = torch.nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        if not use_adam:
            optimizer_ft = torch.optim.SGD(e2cnn_model1.parameters(), lr=learning_rate, momentum=0.9)
        else:
            optimizer_ft = torch.optim.Adam(e2cnn_model1.parameters(), lr=learning_rate)

        # Decay LR by a factor of 0.1 every 7 epochs
        if not use_adam:
            exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=scheduler_step, gamma=0.1)
        else:
            exp_lr_scheduler = None

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
    visualize_model(e2cnn_model1, data_transforms)
    easy_savefig(outfpn='plots/visualize_model.png')
