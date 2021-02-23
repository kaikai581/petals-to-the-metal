#!/usr/bin/env python
'''
This script makes plots to compare the performance of e2cnn and the reference model, wide resnet.
'''

# redirect the output to avoid issues with remote x11
import matplotlib
matplotlib.use('Agg')

import numpy as np
import os
import pandas as pd
import seaborn as sns

class model_comparison:
    '''
    A class to compare two models and make comparison plots.
    '''
    def __init__(self, test_model_name='wrn16_8_stl_d8d4d1', ref_model_name='wide_resnet50_2', model_info_file='processed_data/model_info.csv', prediction_file='processed_data/predictions.csv'):
        '''
        Use string model names to identify model sizes and accuracies.
        '''
        # store the model names
        self.test_model_name = test_model_name
        self.ref_model_name = ref_model_name
        # load basic info table containing model names and model sizes
        self.df_model_info = pd.read_csv(model_info_file)
        # load model prediction table to calculate accuracy
        self.df_model_prediction = pd.read_csv(prediction_file)

        # add accuracy to model information table
        self.add_accuracy_to_model_info()

        # make sure the output folder exists
        if not os.path.exists('plots'): os.makedirs('plots')
    
    def add_accuracy_to_model_info(self):
        '''
        Calculate accuracy for relevant models and add to the model info table.
        Besides, also calculate relative model size and accuracy.
        '''
        # add accuracy to the model info table
        df_acc = pd.DataFrame(columns=['model_name', 'accuracy'])
        for mname in [self.test_model_name, self.ref_model_name]:
            row = dict()
            row['model_name'] = mname
            pred_correctness = pd.Series(np.where(self.df_model_prediction[mname+'_label'] == self.df_model_prediction['true_pytorch_label'], 1, 0)).value_counts()
            row['accuracy'] = pred_correctness[1]/(pred_correctness[1]+pred_correctness[0])
            df_acc = df_acc.append(row, ignore_index = True)
        self.df_model_info = self.df_model_info.merge(df_acc, on='model_name')
        
        # add ratio of model size and accuracy
        for col_name in ['trainable_parameters', 'accuracy']:
            denom = self.df_model_info[self.df_model_info.model_name == self.ref_model_name][col_name].values[0]
            self.df_model_info[col_name+' ratio'] = [val/denom for val in self.df_model_info[col_name]]
    
    def make_model_accuracy_barchart(self):
        '''
        Compare the accuracy of each model.
        '''
        df = self.df_model_info.sort_values('model_name', ascending=True)
        sns_plot = sns.barplot(x='accuracy', y='model_name', data=df)
        
        # add values to the bars
        for p in sns_plot.patches:
            sns_plot.annotate('{:.1%}'.format(p.get_width()),
                    (p.get_width(), p.get_y()+p.get_height()/2),
                    ha = 'center', va = 'center', 
                    xytext = (0, 0), fontweight='bold', color='black',
                    textcoords = 'offset points')
        
        # edit x and y labels
        sns_plot.set(xlabel='accuracy', ylabel='model name')
        
        # modify tick labels
        labels = [item.get_text() for item in sns_plot.get_yticklabels()]
        labels[0] = '(reference model)\nwide_resnet50_2'
        labels[1] = '(e2cnn variant)\nwrn16_8_stl_d8d4d1'
        sns_plot.set_yticklabels(labels)

        sns_plot.figure.tight_layout()
        sns_plot.figure.savefig('plots/model_accuracy_barchart.png')
        sns_plot.figure.clf()

    def make_model_size_barchart(self):
        '''
        Compare the number of parameters in each model.
        '''
        df = self.df_model_info.sort_values('model_name', ascending=True)
        sns_plot = sns.barplot(x='trainable_parameters', y='model_name', data=df)
        
        # add values to the bars
        for p in sns_plot.patches:
            sns_plot.annotate(int(p.get_width()),
                    (p.get_width(), p.get_y()+p.get_height()/2),
                    ha = 'center', va = 'center', 
                    xytext = (0, 0), fontweight='bold', color='black',
                    textcoords = 'offset points')
        
        # edit x and y labels
        sns_plot.set(xlabel='number of model paremeters', ylabel='model name')
        
        # modify tick labels
        labels = [item.get_text() for item in sns_plot.get_yticklabels()]
        labels[0] = '(reference model)\nwide_resnet50_2'
        labels[1] = '(e2cnn variant)\nwrn16_8_stl_d8d4d1'
        sns_plot.set_yticklabels(labels)

        sns_plot.figure.tight_layout()
        sns_plot.figure.savefig('plots/model_size_barchart.png')
        sns_plot.figure.clf()

    def make_ratio_barchart(self):
        '''
        Make bar chart of the model size ratio and accuracy ratio.
        How to make seaborn plots if hue species actually scatter in two columns?
        Ref: https://stackoverflow.com/questions/40877135/plotting-two-columns-of-dataframe-in-seaborn
        '''
        df = self.df_model_info.melt(id_vars='model_name', value_vars=['trainable_parameters ratio', 'accuracy ratio'])
        df = df.sort_values('model_name')
        sns_plot = sns.barplot(x='model_name', y='value', hue='variable', data=df)

        # decorate legend
        leg = sns_plot.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=len(df.model_name.unique()), frameon=False)
        leg.get_texts()[0].set_text('number of model paremeters')
        leg.get_texts()[1].set_text('accuracy')

        # modify tick labels
        labels = [item.get_text() for item in sns_plot.get_xticklabels()]
        labels[0] = '(reference model)\nwide_resnet50_2'
        labels[1] = '(e2cnn variant)\nwrn16_8_stl_d8d4d1'
        sns_plot.set_xticklabels(labels)

        # edit x and y labels
        sns_plot.set(xlabel='model name', ylabel='relative value')

        # add values to the bars
        for p in sns_plot.patches:
            sns_plot.annotate('{:.1%}'.format(p.get_height()),
                    (p.get_x()+p.get_width()/2, p.get_height()),
                    ha = 'center', va = 'center', 
                    xytext = (0, 4), fontweight='bold', color='black',
                    textcoords = 'offset points')

        sns_plot.figure.tight_layout()
        sns_plot.figure.savefig('plots/model_performance_barchart.png')
        sns_plot.figure.clf()

if __name__ == '__main__':
    mcomp = model_comparison()
    mcomp.make_ratio_barchart()
    mcomp.make_model_size_barchart()
    mcomp.make_model_accuracy_barchart()
