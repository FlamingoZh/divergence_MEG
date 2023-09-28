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

def compare_models(D,P1,P2,p1,p2):
    fig,ax = plt.subplots(5,4,sharex=True,sharey=True,figsize=(8, 10))
    fig.tight_layout()
    for t in range(D.shape[0]):
        corr2=[pearsonr(D[t,:,ch], P1[t,:,ch])[0] for ch in range(D.shape[2])]
        corr1=[pearsonr(D[t,:,ch], P2[t,:,ch])[0] for ch in range(D.shape[2])]
        row=t//4
        col=t%4
        # print(row,col)
        
        sig_channels_ft=[ch for ch in range(D.shape[2]) if p1[ch,t]<0.05]
        sig_channels_base=[ch for ch in range(D.shape[2]) if p2[ch,t]<0.05]
        
        ax[row,col].axline((0.5, 0.5), slope=1, ls="--", c=".8", linewidth=0.8)
        
        ax[row,col].scatter(corr1,corr2,s=1.0,c="#D2D2D3")
        ax[row,col].scatter([corr1[item] for item in sig_channels_ft],[corr2[item] for item in sig_channels_ft],s=1.0,c="#EB455F")
        ax[row,col].scatter([corr1[item] for item in sig_channels_base],[corr2[item] for item in sig_channels_base],s=1.0,c="#2B3467")
        ax[row,col].title.set_text(f'{t*25} to {(t+1)*25} ms')
        
        ax[row,col].text(0.15,0.49,f"{np.round(len(sig_channels_ft)/D.shape[2]*100,1)}%",c="#EB455F")
        ax[row,col].text(0.41,0.16,f"{np.round(len(sig_channels_base)/D.shape[2]*100,1)}%",c="#2B3467")
        
        ax[row,col].set_xlim(0.13,0.57)
        ax[row,col].set_ylim(0.13,0.57)
        
        # ax[row,col].set_xlim(0.15,0.55)
        # ax[row,col].set_ylim(0.15,0.55)
        ax[row,col].set_xticks([0.2,0.3,0.4,0.5])
        ax[row,col].set_yticks([0.2,0.3,0.4,0.5])
        ax[row,col].set_aspect('equal')

    fig.text(0.4,0,"Correlation on Base Model")
    fig.text(0,0.4,"Correlation on Finetuned Model",rotation=90)

    return fig

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument("--dataset", default="HP")
	parser.add_argument("--chapter", type=int, default=1)

	parser.add_argument("--base_path", default="/home/yuchen/Desktop/Harry_divergence/interim_data/data_for_analysis/")
	parser.add_argument("--pval_base_path", default="/home/yuchen/Desktop/Harry_divergence/interim_data/permutation/")
	parser.add_argument("--data_base_path")
	parser.add_argument("--data_ft_path")
	parser.add_argument("--pvals_path")
	parser.add_argument("--ft_model_info")

	args = parser.parse_args()

	data_base=pickle.load(open(f"{args.base_path}{args.data_base_path}","rb"))
	data_ft=pickle.load(open(f"{args.base_path}{args.data_ft_path}","rb"))

	pvals=pickle.load(open(f"{args.pval_base_path}{args.pvals_path}","rb"))

	p1_corrected=utils.FDR_correction(pvals) # finetuned better than base
	p2_corrected=utils.FDR_correction(1-pvals) # finetuned worse than base

	fig=compare_models(data_base["actual_all_tw"],data_ft["pred_all_tw"],data_base["pred_all_tw"],p1_corrected,p2_corrected)
	fig.savefig(f'../figures/compare_{args.ft_model_info}_{args.dataset}_chapter_{args.chapter}.png', dpi=1000, bbox_inches='tight')