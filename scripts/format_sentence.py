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

from tqdm import tqdm
from typing import List, Tuple

def first_substr(lst: List[str], search_strings: List[str] = [".", "!", "?"]) -> int:
	return next(i for i, x in enumerate(lst) if any(string in x for string in search_strings))

def last_substr(lst: List[str], search_strings: List[str] = [".", "!", "?"]) -> int:
	rev_lst = lst[::-1]
	ind = first_substr(rev_lst, search_strings)
	return len(lst) - ind - 1

def get_divergent_word_info(words_sorted: List, words: List) -> Tuple[List[dict], List[str]]:
	sentences = []
	for div_word in tqdm(words_sorted):
		_, word, word_ind = div_word

		word_no_punct = "".join(c for c in word if c not in string.punctuation)

		try:
			forward_ind = first_substr(words[word_ind:])
		except StopIteration:
			forward_ind = 0
		try:
			backward_ind = last_substr(words[:word_ind])
		except StopIteration:
			backward_ind = -1
		contain_sent = words[backward_ind + 1:word_ind + forward_ind + 1]
		sentences.append(" ".join(contain_sent))

	return sentences

def save_sents(divergent_sents: List[str], similar_sents: List[str], fname: str) -> None:
	# data_dict = {
	# 	"generation": "",
	# 	"example_hypotheses": [],
	# 	"dataset_description": "sentences containing words that the brain found surprising (high MSE) or unsurprising (low MSE)",
	# 	"target": "what types of sentences the brain tends to find more surprising",
	# 	"user": "a literary analyst examining the language of a novel, given neuroimaging data on what the brain finds surprising",
	# 	"A_desc": "sentences the brain found surprising",
	# 	"B_desc": "sentences the brain found unsurprising",
	# 	"split": {
	# 		"research": {"A_samples": [], "B_samples": []},
	# 		"validation": {"A_samples": [], "B_samples": []}
	# 	}
	# }

	# data_dict = {
	#     "generation": "",
	#     "example_hypotheses": [],
	#     "dataset_description": "sentences containing words that induce similar or divergent reactions in human cognitive processes and language models",
	#     "target": "what type of sentences induce divergent reactions in human cognitive processes compared to language models",
	#     "user": "A researcher conducting a comparative analysis of the divergent behaviors exhibited by human brains and language models.",
	#     "A_desc": "sentences in which humans and language models exhibit divergent responses or interpretations.",
	#     "B_desc": "sentences in which humans and language models exhibit similar responses or interpretations.",
	#     "split": {
	#         "research": {"A_samples": [], "B_samples": []},
	#         "validation": {"A_samples": [], "B_samples": []}
	#     }
	# }


	# data_dict = {
	# 	"dataset_description": "sentences from Harry Potter and the Sorcerer's Stone",
	# 	"generation": "whether humans and language models exhibit divergent or similar responses to those sentences",	    
	# 	"target": "what type of sentences induce different reactions in human cognitive processes and language models",
	# 	"user": "A researcher conducting a comparative analysis of the divergent behaviors exhibited by humans and language models.",
	# 	"A_desc": "sentences in which humans and language models exhibit divergent responses or interpretations",
	# 	"B_desc": "sentences in which humans and language models exhibit similar responses or interpretations",
	# 	"example_hypotheses": ['contain figurative language', 'contain emotions', 'refer to physical objects'],
	# 	"split": {
	# 		"research": {"A_samples": [], "B_samples": []},
	# 		"validation": {"A_samples": [], "B_samples": []}
	# 	}
	# }

	data_dict = {
		"dataset_description": "two chapters from 'Harry Potter and the Sorcerer's Stone'",
		"generation": "the accuracy of language models in predicting human responses to these sentences",	    
		"target": "which sentences pose difficulties for language models when predicting human responses",
		"user": "a literary analyst investigating the characteristics of words that challenge language models in predicting human responses",
		"A_desc": "sentences where language models poorly predict human responses",
		"B_desc": "sentences where language models accurately predict human responses",
		"example_hypotheses": [],
		# "example_hypotheses": ['contain figurative language', 'contain emotions', 'refer to physical objects'],
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

## Get most and least divergent words
def sort_list(words,MSEs):
	# agg_MSEs=np.array([np.mean(MSEs[max(0,i-4):i+1]) for i in range(len(MSEs))])
	agg_MSEs=np.array([MSEs[i] for i in range(len(MSEs))])

	idx=np.argsort(agg_MSEs)
	return [(agg_MSEs[i],words[i],i) for i in idx][::-1]

if __name__ == "__main__":

	home=os.path.expanduser("~")

	parser = argparse.ArgumentParser()
	
	parser.add_argument("--dataset", default="HP")
	parser.add_argument("--sequence_length", type=int, default=100)
	
	parser.add_argument("--save_path", default=f"{home}/Desktop/Harry_divergence/interim_data/divergent_sents/")

	parser.add_argument("--dump", action="store_false", help="dump generated data (by default True)")

	args = parser.parse_args()


	print("Echo arguments:",args)

	## Load data
	data1=pickle.load(open(f"{home}/Desktop/Harry_divergence/interim_data/data_for_analysis/HP_chapter_1_base.pkl","rb"))

	words1=data1['text'][args.sequence_length:]
	MSEs1=np.mean(data1['MSE_all_tw'][4:17],axis=0)

	data2=pickle.load(open(f"{home}/Desktop/Harry_divergence/interim_data/data_for_analysis/HP_chapter_2_base.pkl","rb"))

	words2=data2['text'][args.sequence_length:]
	MSEs2=np.mean(data2['MSE_all_tw'][4:17],axis=0)

	words=words1+words2
	MSEs=np.concatenate([MSEs1,MSEs2])
	# words=words1
	# MSEs=MSEs1
	# print(len(words),MSEs.shape)
	
	words_sorted = sort_list(words,MSEs)
	divergent_sents = get_divergent_word_info(words_sorted[:100], words)
	similar_sents = get_divergent_word_info(words_sorted[-100:], words)

	## Save data
	save_sents(divergent_sents, similar_sents, f"{args.save_path}divergent_sents_chapter_1.pkl")

