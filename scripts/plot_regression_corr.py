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

def compute_corrs(actual, pred):
	corrs=np.zeros((actual.shape[2], actual.shape[0]))
	for t in range(actual.shape[0]):
		for ch in range(actual.shape[2]):
			corrs[ch,t]=pearsonr(actual[t,:,ch],pred[t,:,ch])[0]
	return corrs

def plot_num_of_sig_channels(sig_channels_all_tw,figsize=(20,6),fontsize=16):
	fig, ax = plt.subplots(figsize=figsize)
	num_sig_channels=[len(i) for i in sig_channels_all_tw]
	ax.plot(num_sig_channels,linewidth=5,color='C1')
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.set_xticks(range(20),[f"{time*25}" for time in range(20)],fontsize=fontsize)
	ax.set_yticklabels(num_sig_channels,fontsize=fontsize)
	ax.set_xlabel("Time (ms)",fontsize=fontsize)
	ax.set_ylabel("# of Significant Channels",fontsize=fontsize)

	return fig

if __name__ == '__main__':
	home=os.path.expanduser("~")

	parser = argparse.ArgumentParser()
	
	parser.add_argument("--dataset", default="HP")
	parser.add_argument("--chapter", type=int, default=1)

	parser.add_argument("--base_path", default=f"{home}/Desktop/Harry_divergence/interim_data/data_for_analysis/")
	parser.add_argument("--data_path", default="HP_chapter_1_base.pkl")
	
	parser.add_argument("--model_info")

	args = parser.parse_args()

	data=pickle.load(open(f"{args.base_path}{args.data_path}","rb"))
	pred=data['pred_all_tw']
	actual=data['actual_all_tw']
	all_tw_corrs=compute_corrs(actual,pred)

	figure=meg_sensor_plot.topoplot(all_tw_corrs,vmax=np.max(all_tw_corrs),vmin=np.min(all_tw_corrs), \
		nrow=2, ncol=10, figsize=(20, 6), \
		cmap= 'YlOrBr');
	figure.savefig(f"../figures/{args.dataset}_corr_temporal_{args.model_info}.pdf",format="pdf",bbox_inches='tight')

	figure=plot_num_of_sig_channels(data['sig_channels_all_tw'])
	figure.savefig(f"../figures/{args.dataset}_num_sig_channels_{args.model_info}.pdf",format="pdf",bbox_inches='tight')


