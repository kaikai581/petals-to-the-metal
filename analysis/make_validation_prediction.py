#!/usr/bin/env python

# my own module
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from utilities.helpers import get_git_root

from torchvision import datasets, models, transforms
from utilities.e2wrn import wrn16_8_stl_d8d4d1
import pandas as pd
import torch

# GPU or CPU as a global variable
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class trained_model_prediction:
    '''
    This class applies trained models to predict class labels of images in the validation sample.
    '''
    def __init__(self, model_name, data_dir, img_resize=96):
        '''
        In the constructor the dataloader is constructed.
        '''
        # store the model name
        self.model_name = model_name

        # Data augmentation and normalization for training and validation
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(img_resize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(img_resize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        self.image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
        # Since I have to make correspondence between predicted labels and filename, i have to use batch_size 1 and shuffle False. Otherwies I have to make my own DataLoader.
        # ref: https://stackoverflow.com/questions/56699048/how-to-get-the-filename-of-a-sample-from-a-dataloader
        self.dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=1, shuffle=False, num_workers=4) for x in ['train', 'val']}

    def load_trained_model(self, trained_model):
        self.model = trained_model
        self.model.eval()
    
    def make_output_df(self, overwrite=False):
        '''
        Make and return a dataframe for model predictions.
        '''
        if (not os.path.exists(self.outfpn)) or overwrite:
            data_dict = dict()
            data_dict['filename'] = []
            data_dict['true_original_label'] = []
            data_dict['true_pytorch_label'] = []
            for fpn, label in self.image_datasets['val'].imgs:
                data_dict['filename'].append(os.path.basename(fpn))
                data_dict['true_original_label'].append(os.path.dirname(fpn).split('/')[-1])
                data_dict['true_pytorch_label'].append(label)
            df_prediction = pd.DataFrame(data_dict)
        else:
            df_prediction = pd.read_csv(self.outfpn)

        return df_prediction
    
    def predict_save(self, outfpn, overwrite=False):
        '''
        Predict class labels and save to file.
        If outfpn exists already, load the data into a dataframe and add new data to it.
        '''
        # prediction container
        data_dict = dict()
        data_dict['filename'] = []
        pred_colname = self.model_name+'_label'
        data_dict[pred_colname] = []
        # Iterate over data.
        for i, (images, labels) in enumerate(self.dataloaders['val'], 0):
            images = images.to(device)
            labels = labels.to(device)
            with torch.set_grad_enabled(False):
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                data_dict[pred_colname].extend(preds.tolist())
                sample_fname, _ = self.dataloaders['val'].dataset.samples[i]
                data_dict['filename'].append(os.path.basename(sample_fname))

        # prepare output dataframe
        self.outfpn = outfpn
        self.df_prediction = self.make_output_df(overwrite=overwrite)
        # self.df_prediction.set_index('filename', inplace=True)
        self.df_prediction = self.df_prediction.merge(pd.DataFrame(data_dict), on='filename', how='left')
        
        # save to file
        outdir = os.path.dirname(outfpn)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        self.df_prediction.to_csv(outfpn, index=False)
    
def prepare_base_model(model_fpn, image_datasets):
    '''
    The base model is the wide resnet provided by pytorch.
    Adapt the number of classes to my data.
    '''
    model_ft = models.wide_resnet50_2()
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to len(class_names).
    class_names = image_datasets['train'].classes
    model_ft.fc = torch.nn.Linear(num_ftrs, len(class_names))
    model_ft = model_ft.to(device)
    model_ft.load_state_dict(torch.load(model_fpn))

    return model_ft

if __name__ == '__main__':
    # the data directory
    data_dir = os.path.join(get_git_root(__file__), 'data/imagefolder-jpeg-224x224')

    # load base model
    base_model_prediction = trained_model_prediction('wide_resnet50_2', data_dir, 224)
    base_model_pathname = os.path.join(get_git_root(__file__), 'model_weights/224x224/wide_resnet_no_augmentation.pt')
    base_model = prepare_base_model(base_model_pathname, base_model_prediction.image_datasets)
    base_model_prediction.load_trained_model(base_model)
    base_model_prediction.predict_save('processes_data/predictions.csv', overwrite=True)

    # load e2cnn model
    e2cnn_model_prediction = trained_model_prediction('wrn16_8_stl_d8d4d1', data_dir, 96)
    e2cnn_model_pathname = os.path.join(get_git_root(__file__), 'model_weights/96x96/wrn16_8_stl_d8d4d1_lr1.0e-05_sgd_epoch90.pt')
    e2cnn_model = wrn16_8_stl_d8d4d1(num_classes=len(base_model_prediction.image_datasets['train'].classes))
    e2cnn_model = e2cnn_model.to(device)
    e2cnn_model.load_state_dict(torch.load(e2cnn_model_pathname))
    e2cnn_model_prediction.load_trained_model(e2cnn_model)
    e2cnn_model_prediction.predict_save('processed_data/predictions.csv')