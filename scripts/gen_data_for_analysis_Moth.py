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
from sklearn.decomposition import PCA

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

def construct_stimuli_and_brain_responses(MEG, word_embeddings, word_onset_point, story_dict, args, layer = -1, PCA_n_components = 50, n_delays = 40, do_zscore = True):
	"""construct stimuli and brain responses for regression"""
	
	# construct brain responses
	brain_responses = []
	for story in story_dict:
		avg_MEG = []
		for trial in story_dict[story]:
			if do_zscore:
				avg_MEG.append(zscore(MEG[trial]))
			else:
				avg_MEG.append(MEG[trial])
		brain_responses.append(np.stack(avg_MEG).mean(axis=0))
	brain_responses = np.concatenate(brain_responses, axis = 0)	# (word, channel)

	stimuli = np.zeros((brain_responses.shape[0], word_embeddings.shape[2]))	# (word, embedding_dim)

	# construct stimuli
	offset = 0
	idx = 0
	for i, story in enumerate(story_dict):
		trial = story_dict[story][0]
		for time in word_onset_point[trial][args.sequence_length:]:
			stimuli[time+offset] = word_embeddings[args.layer][idx]
			idx += 1
		offset += MEG[trial].shape[0]

	# reduce dimensionality of stimuli
	pca = PCA(n_components=PCA_n_components).fit(stimuli)
	stimuli = pca.transform(stimuli)	# (word, PCA_n_components)
	
	# add delays
	stimuli = utils.delay_mat(stimuli, range(n_delays))	# (word, PCA_n_components*n_delays)

	return stimuli, brain_responses

if __name__ == '__main__':
	home=os.path.expanduser("~")

	parser = argparse.ArgumentParser()
	
	parser.add_argument("--dataset", default="Moth")
	parser.add_argument("--base_data_path", default=f"{home}/Desktop/MEG_divergence/Moth_data/")
	parser.add_argument("--p_thresh", type=int, default=0.001)

	parser.add_argument("--sequence_length", type=int, default=20)
	parser.add_argument("--lm_embed_path", default=f"{home}/Desktop/MEG_divergence/interim_data/lm_embeddings/Moth_GPT2-xl_base.pkl")
	parser.add_argument("--lm_name", default="GPT2-xl")
	parser.add_argument("--model_info", default="base")
	parser.add_argument("--layer", type = int, default=-1)


	args = parser.parse_args()

	print("Echo arguments:",args)
	
	## Load data

	MEG = pickle.load(open("../Moth_data/Moth_MEG.pkl","rb"))
	word_onset_point = pickle.load(open("../Moth_data/Moth_word_onset_point.pkl","rb"))
	word_embeddings = pickle.load(open(args.lm_embed_path,"rb"))["all_embeddings"]
	story_dict = pickle.load(open("../Moth_data/Moth_run_info_test_story_dict.pkl","rb"))

	# Construct matrix of features

	all_embeddings, meg_data = construct_stimuli_and_brain_responses(MEG, word_embeddings, word_onset_point, story_dict, args)
	all_embeddings = np.expand_dims(all_embeddings, axis=0)
	meg_data = np.expand_dims(meg_data, axis=2)

	print("embedding shape:", all_embeddings.shape)
	print("meg_data shape:", meg_data.shape)

	word_num = meg_data.shape[0]
	channel_num = meg_data.shape[1]
	timepoint_num = 1
	layer_num = 1

	## init arrays
	all_corrs = np.zeros((layer_num, timepoint_num, channel_num))
	all_sig_channels = np.zeros((layer_num, timepoint_num, channel_num), dtype=int)
	all_pred = np.zeros((layer_num, timepoint_num, word_num, channel_num))
	all_MSE = np.zeros((layer_num, timepoint_num, word_num))
	all_cos_sim = np.zeros((layer_num, timepoint_num, word_num))

	## Ridge regression
	print("Ridge regression...")
	for layer in range(layer_num):
		for time in range(timepoint_num):
			brain_responses = meg_data[:,:,time]
			embeddings = all_embeddings[layer]
			brain_responses_pred = utils.do_ridge_regression(embeddings, brain_responses)

			all_pred[layer, time] = brain_responses_pred

			for ch, (pred,actual) in enumerate(zip(brain_responses_pred.T, brain_responses.T)):	# for each channel, all words
				corr, pvalue = pearsonr(pred,actual,alternative="greater")
				all_corrs[layer, time, ch] = corr
				if pvalue < args.p_thresh:
					all_sig_channels[layer, time, ch] = 1

			sig_channels = all_sig_channels[layer, time]
			print(f"layer {args.layer}: {np.sum(sig_channels)}")

			for word, (pred,actual) in enumerate(zip(brain_responses_pred, brain_responses)):	# for each word, all sig channels
				all_MSE[layer, time, word] = np.linalg.norm(pred[sig_channels]-actual[sig_channels], ord=2) ** 2 / len(sig_channels)
				all_cos_sim[layer, time, word] = utils.cosine_similarity(pred[sig_channels],actual[sig_channels])


	pickle.dump(
		dict(
		embedding=all_embeddings,
		actual=meg_data,
		pred=all_pred,
		corr=all_corrs,
		MSE=all_MSE,
		cos=all_cos_sim,
		sig_channels=all_sig_channels,
		),
		open(f"../interim_data/data_for_analysis/{args.dataset}_{args.lm_name}_{args.model_info}.pkl","wb"))
