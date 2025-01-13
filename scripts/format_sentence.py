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

def save_sents(divergent_sents, similar_sents, fname):

	data_dict = {
		"dataset_description": "two chapters from 'Harry Potter and the Sorcerer's Stone'",
		"generation": "the difference between language model and human responses to these sentences",	    
		"target": "which sentences induce different responses for language models and human responses",
		"user": "a literary analyst investigating the characteristics of words",
		"A_desc": "sentences where language models and humans show divergent responses",
		"B_desc": "sentences where language models and humans show similar responses",
		"example_hypotheses": [],
		"split": {
			"research": {"A_samples": [], "B_samples": []},
			"validation": {"A_samples": [], "B_samples": []}
		}
	}

	# train/val split (no test)
	train_split_surp = int(len(divergent_sents) * 0.8)
	train_split_unsurp = int(len(similar_sents) * 0.8)
	divergent_sents_train = divergent_sents[:train_split_surp]
	similar_sents_train = similar_sents[:train_split_unsurp]
	divergent_sents_val = divergent_sents[train_split_surp:]
	similar_sents_val = similar_sents[train_split_unsurp:]

	data_dict["split"]["research"]["A_samples"] = divergent_sents_train
	data_dict["split"]["validation"]["A_samples"] = similar_sents_train
	data_dict["split"]["research"]["B_samples"] = divergent_sents_val
	data_dict["split"]["validation"]["B_samples"] = similar_sents_val

	with open(fname, "wb") as f:
		pickle.dump(data_dict, f)
	
	
def rank_sentences(words, mses, search_strings = [".", "!", "?"]):
	all_sentences = list()
	this_sentence = list()
	mse_all_sentences = list()
	mse_count = 0
	
	for i, word in enumerate(words):
		this_sentence.append(word)
		mse_count += mses[i]
		if any(string in word for string in search_strings) and len(this_sentence) > 3: # end of a sentence
			all_sentences.append(" ".join(this_sentence))
			mse_all_sentences.append(mse_count/len(this_sentence))
			this_sentence = list()
			mse_count = 0
			
	sort_idx = np.argsort(mse_all_sentences)
	return [all_sentences[idx] for idx in sort_idx]

if __name__ == "__main__":

	home=os.path.expanduser("~")

	parser = argparse.ArgumentParser()
	
	parser.add_argument("--dataset", default="HP")
	parser.add_argument("--sequence_length", type=int, default=20)

	parser.add_argument("--chapter", type=int, default=1)

	parser.add_argument("--layer", type=int, default=-1)
	
	parser.add_argument("--save_path", default=f"{home}/Desktop/MEG_divergence/interim_data/divergent_sents/")

	parser.add_argument("--lm_name", default= "GPT2-xl")

	parser.add_argument("--n_sents", type=int, default=100)

	args = parser.parse_args()


	print("Echo arguments:",args)

	## Load data
	if args.chapter==12:
		data1=pickle.load(open(f"{home}/Desktop/MEG_divergence/interim_data/data_for_analysis/HP_chpt1_{args.lm_name}_base.pkl","rb"))
		words1=data1['text'][args.sequence_length:]
		MSEs1=np.mean(data1['MSE'][args.layer][12:17],axis=0)
		data2=pickle.load(open(f"{home}/Desktop/MEG_divergence/interim_data/data_for_analysis/HP_chpt2_{args.lm_name}_base.pkl","rb"))
		words2=data2['text'][args.sequence_length:]
		MSEs2=np.mean(data2['MSE'][args.layer][12:17],axis=0)
		words=words1+words2
		MSEs=np.concatenate([MSEs1,MSEs2])
	elif args.chapter==1:
		data=pickle.load(open(f"{home}/Desktop/MEG_divergence/interim_data/data_for_analysis/HP_chpt1_{args.lm_name}_base.pkl","rb"))
		words=data['text'][args.sequence_length:]
		MSEs=np.mean(data['MSE'][args.layer][12:17],axis=0)
	elif args.chapter==2:
		data=pickle.load(open(f"{home}/Desktop/MEG_divergence/interim_data/data_for_analysis/HP_chpt2_{args.lm_name}_base.pkl","rb"))
		words=data['text'][args.sequence_length:]
		MSEs=np.mean(data['MSE'][args.layer][12:17],axis=0)
	
	ranked_sentences = rank_sentences(words, MSEs)
	similar_sents = ranked_sentences[:args.n_sents]
	divergent_sents = ranked_sentences[-args.n_sents:]
	print("similar sentences:", similar_sents[:5])
	print("divergent sentences:", divergent_sents[-5:])

	## Save data
	save_sents(divergent_sents, similar_sents, f"{args.save_path}formatted_sents_{args.lm_name}_layer{args.layer}_chpt{args.chapter}.pkl")
