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

from transformers import AutoTokenizer, AutoModelForCausalLM

# from mat4py import loadmat

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

sys.path.append("../")
from dataloader.dataloader import HP_dataset_denoised

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

	parser.add_argument("--batch_size", type=int, default=4)
	parser.add_argument("--sequence_length", type=int, default=20)
	parser.add_argument("--lm_name", default="GPT2-xl")
	parser.add_argument("--model_info", default="base")
	parser.add_argument("--lm_path")
	
	args = parser.parse_args()
	args.no_cuda=not torch.cuda.is_available()

	print("Echo arguments:",args)

	## Load dataset
	whole_data = HP_dataset_denoised(args)
	print("Number of words:", len(whole_data))

	## Load LM
	if args.model_info=="base":	# if loading base model
		tokenizer,model=utils.load_tokenizer_and_model_from_transformers(args.lm_name)
	else:	# if loading finetuned model
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
	dataloader=DataLoader(whole_data, batch_size=args.batch_size)

	all_logits=list()
	all_embeddings=list()
	all_last_words=list()
	all_megs=list()

	for batch_i,(batch_text_idx,batch_meg) in enumerate(dataloader):
		batch_text=list()
		batch_last_words=list()
		for text_idx in batch_text_idx:
			line=list()
			for word_idx in text_idx:
				word=whole_data.index_to_word[int(word_idx)]
				line.append(word)
				last_word=word.strip(string.punctuation) # remove punctuations at start and end of the word
			line=" ".join(line)
			# line+=tokenizer.bos_token
			batch_text.append(line)
			batch_last_words.append(last_word)
		encodings = tokenizer(batch_text, return_tensors="pt", padding=True)['input_ids']
		# encodings = tokenizer(batch_text, return_tensors="pt", truncation=True, max_length=args.sequence_length)['input_ids']
		if not args.no_cuda:
			encodings=encodings.to("cuda:0")

		batch_last_word_pos=utils.find_indices_of_last_word_in_batch(tokenizer,batch_last_words,encodings)

		with torch.no_grad():
			output=model(encodings)

			for i,sentence_logits in enumerate(output['logits']):
				word_start,word_end=batch_last_word_pos[i]
				all_logits.append(sentence_logits[word_start-1:word_end].detach().cpu().numpy())
				embeddings=np.vstack([torch.mean(embed[i,word_start:word_end+1],dim=0).detach().cpu().numpy()  for embed in output['hidden_states']])
				all_embeddings.append(embeddings)
				all_last_words.append(encodings[i][word_start:word_end+1].detach().cpu().numpy())
			all_megs.append(batch_meg.detach().cpu().numpy())

		if batch_i%100==0:
			print(f"Batch {batch_i+1}/{len(dataloader)}...")

	all_megs=np.vstack(all_megs)	# (word, meg_channel)
	all_embeddings=np.stack(all_embeddings,axis=1)	# (layer, word, embedding_dim)

	dumped_data=dict(
			all_megs=all_megs,
			all_embeddings=all_embeddings,
			all_last_words=all_last_words,
			all_logits=all_logits
		)

	pickle.dump(dumped_data,
		open(f"{home}/Desktop/MEG_divergence/interim_data/lm_embeddings/{args.dataset}_chpt{args.chapter}_{args.lm_name}_{args.model_info}.pkl","wb"))
