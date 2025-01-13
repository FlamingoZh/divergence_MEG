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
	
	parser.add_argument("--dataset", default="Moth")
	parser.add_argument("--base_data_path", default=f"{home}/Desktop/MEG_divergence/Moth_data/")
	# parser.add_argument("--chapter", type=int, default=1)
	# parser.add_argument("--meg_offset", type=int, default=0)

	parser.add_argument("--batch_size", type=int, default=1)
	parser.add_argument("--sequence_length", type=int, default=20)
	parser.add_argument("--lm_name", default="GPT2-xl")
	parser.add_argument("--model_info", default="base")
	parser.add_argument("--lm_path")
	
	args = parser.parse_args()
	args.no_cuda=not torch.cuda.is_available()

	print("Echo arguments:",args)

	## Load dataset
	Moth_words = pickle.load(open("../Moth_data/Moth_words.pkl","rb"))
	Moth_run_info_test_story_dict = pickle.load(open("../Moth_data/Moth_run_info_test_story_dict.pkl","rb"))

	all_sents = []
	for story in Moth_run_info_test_story_dict:
		st = Moth_run_info_test_story_dict[story][0]
		for i in range(args.sequence_length, len(Moth_words[st])):
			all_sents.append(" ".join(Moth_words[st][i-args.sequence_length:i]).lower())
	print("Number of sentences:", len(all_sents))

	## Load LM
	if args.model_info=="base":	# if loading base model
		tokenizer,model=utils.load_tokenizer_and_model_from_transformers(args.lm_name)
	else:	# if loading finetuned model
		from transformers import AutoTokenizer, AutoModelForCausalLM
		if args.lm_name == "Llama-2":
			tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", truncation_side="left", padding_side="left")
		elif args.lm_name == "GPT2-xl":
			tokenizer = AutoTokenizer.from_pretrained("gpt2-xl", truncation_side="left", padding_side="left")
		else:
			raise ValueError("Not a valid lm_name.")
		model = AutoModelForCausalLM.from_pretrained(args.lm_path, output_hidden_states=True)
		
	tokenizer.pad_token = tokenizer.bos_token

	if not args.no_cuda:
		model=model.to("cuda:0")
	model.eval();

	## Get LM embeddings

	all_embeddings= []

	for i, text in enumerate(all_sents):
		encodings = tokenizer(text, return_tensors="pt", padding=True)['input_ids']
		# encodings = tokenizer(batch_text, return_tensors="pt", truncation=True, max_length=args.sequence_length)['input_ids']
		if not args.no_cuda:
			encodings=encodings.to("cuda:0")

		last_word = text.split(" ")[-1]
		last_word_pos = utils.find_indices_of_last_word_in_batch(tokenizer,[last_word],encodings)
		word_start, word_end = last_word_pos[0]

		with torch.no_grad():
			output=model(encodings)
			embeddings=np.vstack([torch.mean(embed[0, word_start:word_end+1],dim=0).detach().cpu().numpy()  for embed in output['hidden_states']])
			all_embeddings.append(embeddings)

		if i % 100 == 0:
			print(f"Finished {i} out of {len(all_sents)}")

	all_embeddings=np.stack(all_embeddings,axis=1)	# (layer, word, embedding_dim)

	dumped_data=dict(
			all_embeddings=all_embeddings,
		)

	pickle.dump(dumped_data,
		open(f"{home}/Desktop/MEG_divergence/interim_data/lm_embeddings/{args.dataset}_{args.lm_name}_{args.model_info}.pkl","wb"))
