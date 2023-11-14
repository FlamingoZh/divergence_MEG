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

	parser = argparse.ArgumentParser()
	
	parser.add_argument("--dataset", default="HP")
	parser.add_argument("--base_data_path", default=f"{home}/Desktop/Harry_divergence/HP_data/")
	parser.add_argument("--chapter", type=int, default=1)
	parser.add_argument("--meg_offset", type=int, default=0)

	parser.add_argument("--lm_embed_path")
	parser.add_argument("--sequence_length", type=int, default=100)
	parser.add_argument("--model_info", default="base")
	parser.add_argument("--dump", action="store_false", help="dump generated data (by default True)")

	args = parser.parse_args()

	print("Echo arguments:",args)
	
	## Load MEG data
	whole_data = HP_dataset_denoised(args)
	denoised_meg=whole_data.all_megs[args.sequence_length:]

	## Load LM embeddings
	data=pickle.load(open(args.lm_embed_path,"rb"))
	all_embeddings=data["all_embeddings"]

	## Ridge regression
	print("Ridge regression...")
	all_tw_corrs=list()
	all_tw_actual=list()
	all_tw_pred=list()
	all_tw_sig_channels=list()
	all_tw_mse=list()
	all_tw_cos_sim=list()

	for time in range(denoised_meg.shape[2]):
		# if time>0:
		# 	continue
		wt,corrs,_,_,sig_channels=utils.do_ridge_regression_and_compute_correlation(all_embeddings,denoised_meg[:,:,time])
		print(f"{time*25}-{(time+1)*25}ms: {len(sig_channels)}")

		predicted_meg=all_embeddings.dot(wt)
		
		MSEs=list()
		cos_sim=list()
		for pred,actual in zip(predicted_meg,denoised_meg[:,:,time]):
			MSEs.append(np.linalg.norm(pred[sig_channels]-actual[sig_channels], ord=2) ** 2)
			cos_sim.append(utils.cosine_similarity(pred,actual))
		
		all_tw_corrs.append(corrs)
		all_tw_actual.append(denoised_meg[:,:,time])
		all_tw_pred.append(predicted_meg)
		all_tw_sig_channels.append(sig_channels)
		all_tw_mse.append(MSEs)
		all_tw_cos_sim.append(cos_sim)

	all_tw_corrs=np.vstack(all_tw_corrs)
	all_tw_actual=np.stack(all_tw_actual,axis=0)
	all_tw_pred=np.stack(all_tw_pred,axis=0)
	all_tw_mse=np.stack(all_tw_mse,axis=0)
	all_tw_cos_sim=np.stack(all_tw_cos_sim,axis=0)

	if args.dump:
		pickle.dump(
			dict(
			embedding=all_embeddings,
			text=whole_data.all_texts,
			MSE_all_tw=all_tw_mse,
			cos_sim_all_tw=all_tw_cos_sim,
			pred_all_tw=all_tw_pred,
			actual_all_tw=all_tw_actual,
			sig_channels_all_tw=all_tw_sig_channels,
			),
			open(f"../interim_data/data_for_analysis/{args.dataset}_chapter_{args.chapter}_{args.model_info}.pkl","wb"))
