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

def plot_stacked_bar_plot(p1,p2):
	fig,ax = plt.subplots(figsize=(10, 2))
	percent_better=np.array([sum(p1[ch,t]<0.05 for ch in range(p1.shape[0]))/p1.shape[0] for t in range(p1.shape[1])])
	percent_worse=np.array([sum(p2[ch,t]<0.05 for ch in range(p2.shape[0]))/p2.shape[0] for t in range(p2.shape[1])])
	percent_nosig=1-percent_better-percent_worse

	print(percent_worse, percent_nosig, percent_better)

	ax.bar(range(p1.shape[1]), percent_worse, color='#2B3467', label="worse")
	ax.bar(range(p1.shape[1]), percent_nosig, bottom=percent_worse, color='#D2D2D3', label="non significant")
	ax.bar(range(p1.shape[1]), percent_better, bottom=percent_worse+percent_nosig, color='#EB455F', label="better")

	ax.set_xticks(range(20),[f"{time*25}" for time in range(20)]);
	ax.legend(bbox_to_anchor =(1.23,-0.02), loc='lower right')
	fig.text(0.4,0,"Time (ms)")
	fig.text(0,0.25,"% of Channels",rotation=90)

	fig.tight_layout()
	fig.savefig(f'../figures/stacked_compare_{args.ft_model_info}_{args.dataset}_chapter_{args.chapter}.png', dpi=1000, bbox_inches='tight')

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	
	parser.add_argument("--dataset", default="HP")
	parser.add_argument("--chapter", type=int, default=1)

	parser.add_argument("--base_path", default="/home/yuchen/Desktop/Harry_divergence/interim_data/permutation/")
	parser.add_argument("--pvals_path")
	parser.add_argument("--ft_model_info")

	args = parser.parse_args()
	
	pvals=pickle.load(open(f"{args.base_path}{args.pvals_path}","rb"))

	p1_corrected=utils.FDR_correction(pvals) # finetuned better than base
	p2_corrected=utils.FDR_correction(1-pvals) # finetuned worse than base

	plot_stacked_bar_plot(p1_corrected,p2_corrected)

