import os
import sys
import numpy as np
import string
import pickle
import argparse

import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from scipy.stats import chi2
from scipy.stats import zscore
import scipy.io as sio
from scipy.io import loadmat

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import statsmodels.api as sm

# from mat4py import loadmat

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

sys.path.append("../")
from dataloader.dataloader import HP_dataset_denoised
from utils import meg_sensor_plot
from utils import utils
from utils.ridge.ridge import bootstrap_ridge

np.set_printoptions(precision=3,suppress=True)

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    home=os.path.expanduser("~")

    data=pickle.load(open(f"{home}/Desktop/Harry_divergence/interim_data/data_for_analysis/HP_chapter_1_base.pkl","rb"))
    
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis('off')
    plt.plot(data['actual_all_tw'][0,0,:50],color="#ed6a5a",linewidth=4)
    fig.savefig(f'../figures/demo_actual_MEG1.png', bbox_inches='tight')
    
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis('off')    
    plt.plot(data['actual_all_tw'][0,0,50:100],color="#f4f1bb",linewidth=4)
    fig.savefig(f'../figures/demo_actual_MEG2.png', bbox_inches='tight')
    
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis('off')    
    plt.plot(data['actual_all_tw'][0,0,100:150],color="#9bc1bc",linewidth=4)
    fig.savefig(f'../figures/demo_actual_MEG3.png', bbox_inches='tight')
    
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis('off')
    plt.plot(data['pred_all_tw'][0,0,:50],color="#ed6a5a",linewidth=4)
    fig.savefig(f'../figures/demo_pred_MEG1.png', bbox_inches='tight')
    
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis('off')    
    plt.plot(data['pred_all_tw'][0,0,50:100],color="#f4f1bb",linewidth=4)
    fig.savefig(f'../figures/demo_pred_MEG2.png', bbox_inches='tight')
    
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis('off')    
    plt.plot(data['pred_all_tw'][0,0,100:150],color="#9bc1bc",linewidth=4)
    fig.savefig(f'../figures/demo_pred_MEG3.png', bbox_inches='tight')