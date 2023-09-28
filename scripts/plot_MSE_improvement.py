import os
import sys
import random
import numpy as np
import string
import pickle
import argparse

import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

from scipy.stats import pearsonr
from scipy.stats import chi2
from scipy.stats import zscore
from scipy.stats import ttest_ind
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

def majority_vote(data):
	result=list()
	for col in data.T:
		if np.sum(col)>len(col)/2:
		# if np.sum(col)>0:
			result.append(1)
		else:
			result.append(0)
	return np.array(result)

def get_mse_by_category(data,annot,offset=100):
	mse_cat=list()
	mse_non_cat=list()
	for i in range(len(annot[offset:])):
		if annot[i+offset]==0:
			mse_non_cat.append(data['MSE_all_tw'][:,i])
		else:
			mse_cat.append(data['MSE_all_tw'][:,i])
	mse_cat=np.array(mse_cat)
	mse_non_cat=np.array(mse_non_cat)
	return mse_cat,mse_non_cat

def plot_MSE_improvement(data1,data2,category,pval):
	fig,ax = plt.subplots(figsize=(4, 3))
	df=list()
	for row in data1:
		for t in range(len(row)):
			df.append([row[t],int(t*25),category])
	for row in data2:
		for t in range(len(row)):
			df.append([row[t],int(t*25),"non-"+category])
	
	df=pd.DataFrame(
		df, columns=['MSE Improvement', 'Time (ms)', 'Word Category']
	)
	colors = ["#EB455F", "grey"]

	g=sns.lineplot(data=df, x="Time (ms)", y="MSE Improvement", hue="Word Category", errorbar="se", palette = colors)
	sns.despine()
	if category=="physical":
		g.legend(loc='lower left', bbox_to_anchor=(0.47, 0.75))
	else:
		g.legend(loc='lower left', bbox_to_anchor=(0.47, 0.05))


	y_bot,y_top=ax.get_ylim()
	for t in range(len(pval)):
		if pval[t]<0.05:
			ax.text(t*25,y_bot,"*")

	# ax.set_xticks(range(20),[f"{time*25}" for time in range(20)]);
	# ax.

	fig.tight_layout()
	fig.savefig(f'../figures/MSE_improve_{category}_{args.dataset}_chapter_{args.chapter}.png', dpi=1000, bbox_inches='tight')

def check_diff_is_sig(data1,data2):
	pval=ttest_ind(data1,data2,axis=0,alternative="greater").pvalue
	return pval

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", default="HP")
	parser.add_argument("--chapter", type=int, default=1)
	parser.add_argument("--base_path", default="/home/yuchen/Desktop/Harry_divergence/interim_data/data_for_analysis/")

	args = parser.parse_args()

	emotion_agree_mat,fig_agree_mat,physical_agree_mat=pickle.load(open("../interim_data/agree_mat.pkl","rb"))
	emotion_annot=majority_vote(emotion_agree_mat)
	fig_annot=majority_vote(fig_agree_mat)
	physical_annot=majority_vote(physical_agree_mat)

	data=pickle.load(open(f"{args.base_path}HP_chapter_1_base.pkl","rb"))
	data_emotion=pickle.load(open(f"{args.base_path}HP_chapter_1_ft_emotion_epoch_4.pkl","rb"))
	data_fig=pickle.load(open(f"{args.base_path}HP_chapter_1_ft_figurative_epoch_7.pkl","rb"))
	data_physical=pickle.load(open(f"{args.base_path}HP_chapter_1_ft_physical_epoch_19.pkl","rb"))

	mse_emotion_base,mse_non_emotion_base=get_mse_by_category(data,emotion_annot)
	mse_emotion_ft,mse_non_emotion_ft=get_mse_by_category(data_emotion,emotion_annot)
	pval_emotion=check_diff_is_sig(mse_emotion_base-mse_emotion_ft, mse_non_emotion_base-mse_non_emotion_ft)
	plot_MSE_improvement(mse_emotion_base-mse_emotion_ft, mse_non_emotion_base-mse_non_emotion_ft,"emotional",pval_emotion)

	mse_fig_base,mse_non_fig_base=get_mse_by_category(data,fig_annot)
	mse_fig_ft,mse_non_fig_ft=get_mse_by_category(data_fig,fig_annot)
	pval_fig=check_diff_is_sig(mse_fig_base-mse_fig_ft, mse_non_fig_base-mse_non_fig_ft)
	plot_MSE_improvement(mse_fig_base-mse_fig_ft, mse_non_fig_base-mse_non_fig_ft,"figurative",pval_fig)

	mse_physical_base,mse_non_physical_base=get_mse_by_category(data,physical_annot)
	mse_physical_ft,mse_non_physical_ft=get_mse_by_category(data_physical,physical_annot)
	pval_physical=check_diff_is_sig(mse_physical_base-mse_physical_ft, mse_non_physical_base-mse_non_physical_ft)
	plot_MSE_improvement(mse_physical_base-mse_physical_ft, mse_non_physical_base-mse_non_physical_ft, "physical", pval_physical)





