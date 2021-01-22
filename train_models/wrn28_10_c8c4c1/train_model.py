#!/usr/bin/env python
'''
This script trains one of the best performing networks from the paper, wrn28 without reflection symmetry.
'''

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from torchvision import transforms
from utilities.e2wrn import wrn28_10_c8c4c1
from utilities.helpers import *
import pickle

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

    # print device information
    print('Device being used:', torch.cuda.get_device_name(0))

    # preview several images
    visualize_images(data_transforms)
    print('Some image previews are produced.')

    # It turns out constructing a e2cnn model takes a lot of time. (~10 minutes)
    # Therefore, a prepared model skeleton is saved to file.
    mod_skl_fn = 'model_skeleton/wrn28_10_c8c4c1.pickle'
    if os.path.exists(mod_skl_fn):
        print('Loading model skeleton...')
        with open(mod_skl_fn, 'rb') as f:
            e2cnn_model1 = pickle.load(f)
    else:
        # construct the model
        print('Constructing the model...')
        e2cnn_model1 = wrn28_10_c8c4c1(num_classes=get_nclasses(data_transforms))
        outdir = os.path.dirname(mod_skl_fn)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        with open(mod_skl_fn, 'wb') as f:
            pickle.dump(e2cnn_model1, f)
    e2cnn_model1 = e2cnn_model1.to(device)
        
