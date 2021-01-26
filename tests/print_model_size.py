#!/usr/bin/env python

import os, sys

from torchvision.transforms.transforms import Resize
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from torchvision import transforms

from utilities.e2wrn import *
from utilities.helpers import *
import argparse
import pandas as pd

if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resize_pixel', type=int, default=96)
    args = parser.parse_args()
    img_size = args.resize_pixel

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
    nclasses = get_nclasses(data_transforms)

    models = dict()
    models['wrn16_8_stl_d8d4d1'] = wrn16_8_stl_d8d4d1(num_classes=nclasses)
    models['wrn16_8_stl_d8d4d4'] = wrn16_8_stl_d8d4d4(num_classes=nclasses)
    models['wrn16_8_stl_d1d1d1'] = wrn16_8_stl_d1d1d1(num_classes=nclasses)
    models['wrn28_10_d8d4d1'] = wrn28_10_d8d4d1(num_classes=nclasses)
    models['wrn28_7_d8d4d1'] = wrn28_7_d8d4d1(num_classes=nclasses)
    models['wrn28_10_c8c4c1'] = wrn28_10_c8c4c1(num_classes=nclasses)
    models['wrn28_10_d1d1d1'] = wrn28_10_d1d1d1(num_classes=nclasses)
    
    model_ft = torchvision.models.wide_resnet50_2(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to len(class_names).
    model_ft.fc = torch.nn.Linear(num_ftrs, nclasses)
    models['wide_resnet50_2'] = torchvision.models.wide_resnet50_2()

    df_summary = pd.DataFrame(columns=['model_name', 'trainable_parameters', 'all_parameters'])
    for name, model in models.items():
        row = [name, model_size_trainable(model), model_size_all(model)]
        df_summary.loc[len(df_summary)] = row
    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df_summary)

    # This script is time-consuming. Therefore save the results.
    easy_savedataframe(df_summary, 'output/{}x{}/model_sizes.csv'.format(img_size, img_size))
