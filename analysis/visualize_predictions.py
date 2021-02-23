#!/usr/bin/env python

# my own modules
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from utilities.helpers import get_git_root, imshow
from make_validation_prediction import trained_model_prediction

# redirect the output to avoid issues with remote x11
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

class visualize_predictions:
    '''
    A class to add new columns to the prediction dataframe for easy visualization and make plots.
    '''
    def __init__(self, data_fpn, ref_imgsize=224, test_imgsize=96):
        '''
        Constructor to load prediction results into a dataframe.
        '''
        # load data into a dataframe
        self.df_predictions = pd.read_csv(data_fpn)
        # get all label column names whose labels are predicted by models
        self.label_cols = [col for col in self.df_predictions.columns.tolist() if col.endswith('_label') and (col not in ['true_original_label', 'true_pytorch_label'])]
        # For each label column predicted by model, add a corresponding column indicating the correctness of prediction.
        self.add_correctness_cols()

        # get the correctness columns
        self.col_correctness = [col for col in self.df_predictions.columns.tolist() if col.endswith(' correctness')]

        # the data directory
        data_dir = os.path.join(get_git_root(__file__), 'data/imagefolder-jpeg-224x224')

        # get dataloaders for reference dataloader and test dataloader
        base_model_prediction = trained_model_prediction('wide resnet', data_dir, ref_imgsize)
        self.base_dataloader = base_model_prediction.dataloaders['val']

        e2cnn_model_prediction = trained_model_prediction('e2cnn', data_dir, test_imgsize)
        self.e2cnn_dataloader = e2cnn_model_prediction.dataloaders['val']


    def add_correctness_cols(self):
        '''
        For each label column predicted by model, add a corresponding column indicating the correctness of prediction.
        '''
        df = self.df_predictions
        for col_name in self.label_cols:
            new_col_name = col_name.rstrip('_label') + ' correctness'
            df[new_col_name] = np.where(df[col_name] == df['true_pytorch_label'], 1, 0)
    
    def get_correct_image(self, label, dataset, random_state=0):
        '''
        Given a label, return an image with label as its true label.
        '''
        df = self.df_predictions[self.df_predictions.true_pytorch_label == label].sample(1, random_state=random_state)
        subset = torch.utils.data.Subset(dataset, df.index)
        loader = torch.utils.data.DataLoader(subset)
        for img, lbl in loader:
            return img
    
    def plot_correctness_matrix(self):
        '''
        Plot a 2D histogram where each axis represents the correctness of one model prediction.
        '''
        sns_plot = sns.histplot(data=self.df_predictions, x=self.col_correctness[1], y=self.col_correctness[0], bins=2, discrete=(True, True), cbar=True)
        # set x and y ticks to only integer values
        sns_plot.yaxis.get_major_locator().set_params(integer=True)
        sns_plot.xaxis.get_major_locator().set_params(integer=True)
        
        # get patches and thus bin counts
        # The easiest way I found to plot with seaborn for eyecandy, unfortunately, is to do this redundant calculation...
        h, xedges, yedges = np.histogram2d(x=self.df_predictions[self.col_correctness[1]], y=self.df_predictions[self.col_correctness[0]], bins=2)
        for i in range(2):
            for j in range(2):
                sns_plot.text(xedges[i]+i/2,yedges[j]+j/2, int(h[i,j]), color='w', ha='center', va='center', fontweight='bold')

        # edit axes labels
        sns_plot.set(xlabel='(e2cnn variant) wrn16_8_stl_d8d4d1 correctness', ylabel='(reference model) wide_resnet50_2 correctness')

        sns_plot.figure.savefig('plots/correctness_matrix.png')
        sns_plot.figure.clf()
    
    def plot_quadrant1(self, nsample=10, random_state=0):
        '''
        Sample some images from quadrant 1 and plot.
        '''
        sel_row = (self.df_predictions[self.col_correctness[1]] == 1) & (self.df_predictions[self.col_correctness[0]] == 1)
        out_subdir = 'both_right'
        
        df_q = self.df_predictions[sel_row]
        df_q = df_q.sample(nsample, random_state=random_state)
        
        # check if output directory exists
        outdir = os.path.join('plots/images', out_subdir)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # select relevant entries from reference and test dataloaders
        ref_subset = torch.utils.data.Subset(self.base_dataloader.dataset, df_q.index)
        test_subset = torch.utils.data.Subset(self.e2cnn_dataloader.dataset, df_q.index)
        ref_subloader = torch.utils.data.DataLoader(ref_subset)
        test_subloader = torch.utils.data.DataLoader(test_subset)
        for (img1, lbl1), (img2, lbl2), (idx, row) in zip(ref_subloader, test_subloader, df_q.iterrows()):
            # add a super title
            fig = plt.figure()
            ax1 = plt.subplot(1, 2, 1)
            ax1.axis('off')
            imshow(img1.cpu().data[0], 'wide resnet\npredicted: {}'.format(row['wide_resnet50_2_label']))
            ax2 = plt.subplot(1, 2, 2)
            ax2.axis('off')
            imshow(img2.cpu().data[0], 'e2cnn\npredicted: {}'.format(row['wrn16_8_stl_d8d4d1_label']))
            fig.suptitle('true label: {}'.format(row['true_pytorch_label']))
            plt.savefig(os.path.join(outdir, row['filename']))
            plt.close()
    
    def plot_quadrant2(self, nsample=10, random_state=0):
        '''
        Same some images from quadrant 2 and plot.
        '''
        # select data of interest
        sel_row = (self.df_predictions[self.col_correctness[1]] == 0) & (self.df_predictions[self.col_correctness[0]] == 1)
        out_subdir = 'only_wide_resnet_right'
        df_q = self.df_predictions[sel_row]
        df_q = df_q.sample(nsample, random_state=random_state)

        # check if output directory exists and create if not
        outdir = os.path.join('plots/images', out_subdir)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        
        # select relevant entries from reference and test dataloaders
        ref_subset = torch.utils.data.Subset(self.base_dataloader.dataset, df_q.index)
        test_subset = torch.utils.data.Subset(self.e2cnn_dataloader.dataset, df_q.index)
        ref_subloader = torch.utils.data.DataLoader(ref_subset)
        test_subloader = torch.utils.data.DataLoader(test_subset)
        for (img1, lbl1), (img2, lbl2), (idx, row) in zip(ref_subloader, test_subloader, df_q.iterrows()):
            # add a super title
            fig = plt.figure()
            ax1 = plt.subplot(2, 2, 1)
            ax1.axis('off')
            imshow(img1.cpu().data[0], 'wide resnet\npredicted: {}'.format(row['wide_resnet50_2_label']))
            ax2 = plt.subplot(2, 2, 2)
            ax2.axis('off')
            imshow(img2.cpu().data[0], 'e2cnn\npredicted: {}'.format(row['wrn16_8_stl_d8d4d1_label']))
            img4 = self.get_correct_image(row['wrn16_8_stl_d8d4d1_label'], self.e2cnn_dataloader.dataset, random_state=random_state)
            ax4 = plt.subplot(2, 2, 4)
            ax4.axis('off')
            imshow(img4.cpu().data[0], '{} should look like:'.format(row['wrn16_8_stl_d8d4d1_label']))
            fig.suptitle('true label: {}'.format(row['true_pytorch_label']))
            plt.savefig(os.path.join(outdir, row['filename']))
            plt.close()
    
    def plot_quadrant3(self, same_prediction=False, nsample=10, random_state=0):
        '''
        Same some images from quadrant 4 and plot.
        '''
        # select data of interest
        df_q = self.df_predictions.copy()
        df_q['same_prediction'] = np.where(df_q[self.label_cols[0]] == df_q[self.label_cols[1]], True, False)
        if same_prediction:
            df_q = df_q[df_q.same_prediction == True]
            out_subdir = 'both_wrong/same_prediction'
        else:
            df_q = df_q[df_q.same_prediction == False]
            out_subdir = 'both_wrong/different_prediction'
        sel_row = (df_q[self.col_correctness[1]] == 0) & (df_q[self.col_correctness[0]] == 0)
        df_q = df_q[sel_row]
        df_q = df_q.sample(nsample, random_state=random_state)

        # check if output directory exists and create if not
        outdir = os.path.join('plots/images', out_subdir)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        
        # select relevant entries from reference and test dataloaders
        ref_subset = torch.utils.data.Subset(self.base_dataloader.dataset, df_q.index)
        test_subset = torch.utils.data.Subset(self.e2cnn_dataloader.dataset, df_q.index)
        ref_subloader = torch.utils.data.DataLoader(ref_subset)
        test_subloader = torch.utils.data.DataLoader(test_subset)
        for (img1, lbl1), (img2, lbl2), (idx, row) in zip(ref_subloader, test_subloader, df_q.iterrows()):
            # add a super title
            fig = plt.figure()
            img0 = self.get_correct_image(row['true_pytorch_label'], self.base_dataloader.dataset, random_state=random_state)
            ax0 = plt.subplot(2, 3, 1)
            ax0.axis('off')
            imshow(img0.cpu().data[0], '{} should look like:'.format(row['true_pytorch_label']))
            ax1 = plt.subplot(2, 3, 2)
            ax1.axis('off')
            imshow(img1.cpu().data[0], 'wide resnet\npredicted: {}'.format(row['wide_resnet50_2_label']))
            ax2 = plt.subplot(2, 3, 3)
            ax2.axis('off')
            imshow(img2.cpu().data[0], 'e2cnn\npredicted: {}'.format(row['wrn16_8_stl_d8d4d1_label']))
            img3 = self.get_correct_image(row['wide_resnet50_2_label'], self.base_dataloader.dataset, random_state=random_state)
            ax3 = plt.subplot(2, 3, 5)
            ax3.axis('off')
            imshow(img3.cpu().data[0], '{} should look like:'.format(row['wide_resnet50_2_label']))
            img4 = self.get_correct_image(row['wrn16_8_stl_d8d4d1_label'], self.e2cnn_dataloader.dataset, random_state=random_state+1)
            ax4 = plt.subplot(2, 3, 6)
            ax4.axis('off')
            imshow(img4.cpu().data[0], '{} should look like:'.format(row['wrn16_8_stl_d8d4d1_label']))
            fig.suptitle('true label: {}'.format(row['true_pytorch_label']))
            plt.savefig(os.path.join(outdir, row['filename']))
            plt.close()

    def plot_quadrant4(self, nsample=10, random_state=0):
        '''
        Same some images from quadrant 4 and plot.
        '''
        # select data of interest
        sel_row = (self.df_predictions[self.col_correctness[1]] == 1) & (self.df_predictions[self.col_correctness[0]] == 0)
        out_subdir = 'only_e2cnn_right'
        df_q = self.df_predictions[sel_row]
        df_q = df_q.sample(nsample, random_state=random_state)

        # check if output directory exists and create if not
        outdir = os.path.join('plots/images', out_subdir)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        
        # select relevant entries from reference and test dataloaders
        ref_subset = torch.utils.data.Subset(self.base_dataloader.dataset, df_q.index)
        test_subset = torch.utils.data.Subset(self.e2cnn_dataloader.dataset, df_q.index)
        ref_subloader = torch.utils.data.DataLoader(ref_subset)
        test_subloader = torch.utils.data.DataLoader(test_subset)
        for (img1, lbl1), (img2, lbl2), (idx, row) in zip(ref_subloader, test_subloader, df_q.iterrows()):
            # add a super title
            fig = plt.figure()
            ax1 = plt.subplot(2, 2, 1)
            ax1.axis('off')
            imshow(img1.cpu().data[0], 'wide resnet\npredicted: {}'.format(row['wide_resnet50_2_label']))
            ax2 = plt.subplot(2, 2, 2)
            ax2.axis('off')
            imshow(img2.cpu().data[0], 'e2cnn\npredicted: {}'.format(row['wrn16_8_stl_d8d4d1_label']))
            img3 = self.get_correct_image(row['wide_resnet50_2_label'], self.base_dataloader.dataset, random_state=random_state)
            ax3 = plt.subplot(2, 2, 3)
            ax3.axis('off')
            imshow(img3.cpu().data[0], '{} should look like:'.format(row['wide_resnet50_2_label']))
            fig.suptitle('true label: {}'.format(row['true_pytorch_label']))
            plt.savefig(os.path.join(outdir, row['filename']))
            plt.close()

if __name__ == '__main__':
    vis_preds = visualize_predictions('processed_data/predictions.csv')
    vis_preds.plot_correctness_matrix()
    vis_preds.plot_quadrant1()
    vis_preds.plot_quadrant2()
    vis_preds.plot_quadrant4()
    vis_preds.plot_quadrant3(same_prediction=True)
    vis_preds.plot_quadrant3(same_prediction=False)