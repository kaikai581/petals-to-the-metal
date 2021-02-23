#!/usr/bin/env python

# my own modules
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from utilities.helpers import get_git_root

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class learning_curve:
    '''
    A class to load data for plotting and saving learning curves.
    '''
    def __init__(self, infpn_train, infpn_validation):
        '''
        Provide a full pathname to load the learning curve data into dataframes.
        '''
        self.ax_loss = None
        self.ax_acc = None
        self.df_train = pd.read_csv(infpn_train)
        self.df_validation = pd.read_csv(infpn_validation)
    
    def plot_learning_curves(self):
        '''
        Plot the learning curves.
        '''
        # plot loss curve
        self.fig_loss = plt.figure(1)
        self.ax_loss = sns.lineplot(x='epoch', y='loss', data=self.df_train, label='train')
        sns.lineplot(x='epoch', y='loss', data=self.df_validation, ax=self.ax_loss, label='validation')

        # plot accuracy curve
        self.fig_acc = plt.figure(2)
        self.ax_acc = sns.lineplot(x='epoch', y='accuracy', data=self.df_train, label='train')
        sns.lineplot(x='epoch', y='accuracy', data=self.df_validation, ax=self.ax_acc, label='validation')
        self.ax_acc.set(ylim=(0, 1.03))
    
    def savefig(self, outfpn):
        '''
        Save learning curves to file.
        '''
        if not self.ax_loss or not self.ax_acc:
            self.plot_learning_curves()
        
        outdir = os.path.dirname(outfpn)
        fname = os.path.basename(outfpn)
        if outdir and not os.path.exists(outdir):
            os.makedirs(outdir)
        self.ax_loss.figure.savefig(os.path.join(outdir, 'loss_{}'.format(fname)))
        self.ax_acc.figure.savefig(os.path.join(outdir, 'accuracy_{}'.format(fname)))

        # clear all figures
        plt.close(self.fig_loss)
        plt.close(self.fig_acc)


if __name__ == '__main__':
    # create a curve instance for wide resnet without data augmentation
    train_log = os.path.join(get_git_root(__file__), 'train_models/wide-resnet/no-augmentation/epoch_info/train_epoch_info.csv')
    validation_log = os.path.join(get_git_root(__file__), 'train_models/wide-resnet/no-augmentation/epoch_info/validation_epoch_info.csv')
    lc_wrn_no_aug = learning_curve(train_log, validation_log)
    lc_wrn_no_aug.savefig('plots/lc_wrn_no_aug.png')

    # create a curve instance for e2cnn wide resnet wrn16_8_stl_d8d4d1 with weight decay and without data augmentation
    train_log = os.path.join(get_git_root(__file__), 'train_models/wrn16_8_stl_d8d4d1_weight_decay/epoch_info/96x96/train_epoch_info_wd5e-4.csv')
    validation_log = os.path.join(get_git_root(__file__), 'train_models/wrn16_8_stl_d8d4d1_weight_decay/epoch_info/96x96/validation_epoch_info_wd5e-4.csv')
    lc_wrn16_8_stl_d8d4d1 = learning_curve(train_log, validation_log)
    lc_wrn16_8_stl_d8d4d1.savefig('plots/lc_wrn16_8_stl_d8d4d1_wd5e-4.png')