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
	parser.add_argument("--base_data_path", default=f"{home}/Desktop/MEG_divergence/HP_data/")
	parser.add_argument("--chapter", type=int, default=1)
	parser.add_argument("--meg_offset", type=int, default=0)
	parser.add_argument("--p_thresh", type=int, default=0.001)

	parser.add_argument("--sequence_length", type=int, default=20)
	parser.add_argument("--lm_name", default="GPT2-xl")
	parser.add_argument("--model_info", default="base")
	parser.add_argument("--layer", type=int, default=-1)


	args = parser.parse_args()

	args.lm_embed_path = f"{home}/Desktop/MEG_divergence/interim_data/lm_embeddings/{args.dataset}_chpt{args.chapter}_{args.lm_name}_{args.model_info}.pkl"

	print("Echo arguments:",args)
	
	## Load MEG data
	whole_data = HP_dataset_denoised(args)
	meg_data=whole_data.all_megs[args.sequence_length:]	# (word, meg_channel, time)

	## Load LM embeddings
	data=pickle.load(open(args.lm_embed_path,"rb"))
	all_embeddings=data["all_embeddings"]	# (layer, word, embedding_dim)

	word_num = meg_data.shape[0]
	channel_num = meg_data.shape[1]
	timepoint_num = meg_data.shape[2]
	layer_num = 1

	## init arrays
	all_corrs = np.zeros((layer_num, timepoint_num, channel_num))
	all_sig_channels = np.zeros((layer_num, timepoint_num, channel_num), dtype=int)
	all_pred = np.zeros((layer_num, timepoint_num, word_num, channel_num))
	all_MSE = np.zeros((layer_num, timepoint_num, word_num))
	all_cos_sim = np.zeros((layer_num, timepoint_num, word_num))

	## Ridge regression
	print("Ridge regression...")
	for time in range(timepoint_num):
		brain_responses = meg_data[:,:,time]
		embeddings = all_embeddings[args.layer]
		brain_responses_pred = utils.do_ridge_regression(embeddings, brain_responses)

		all_pred[0, time] = brain_responses_pred

		for ch, (pred,actual) in enumerate(zip(brain_responses_pred.T, brain_responses.T)):	# for each channel, all words
			corr, pvalue = pearsonr(pred,actual,alternative="greater")
			all_corrs[0, time, ch] = corr
			if pvalue < args.p_thresh:
				all_sig_channels[0, time, ch] = 1

		sig_channels = all_sig_channels[0, time]
		print(f"layer {args.layer}, {time*25}-{(time+1)*25}ms: {np.sum(sig_channels)}")

		for word, (pred,actual) in enumerate(zip(brain_responses_pred, brain_responses)):	# for each word, all sig channels
			all_MSE[0, time, word] = np.linalg.norm(pred[sig_channels]-actual[sig_channels], ord=2) ** 2 / len(sig_channels)
			all_cos_sim[0, time, word] = utils.cosine_similarity(pred[sig_channels],actual[sig_channels])


	pickle.dump(
		dict(
		embedding=all_embeddings,
		text=whole_data.all_texts,
		actual=meg_data,
		pred=all_pred,
		corr=all_corrs,
		MSE=all_MSE,
		cos=all_cos_sim,
		sig_channels=all_sig_channels,
		),
		open(f"../interim_data/data_for_analysis/{args.dataset}_chpt{args.chapter}_{args.lm_name}_{args.model_info}_layer{args.layer}.pkl","wb"))
