import os
import sys
import time
import re
import string
import pickle
import copy

import numpy as np
from numpy.linalg import inv

import torch
# import pandas as pd
from collections import Counter

from scipy.stats import zscore
import scipy.io as sio

class HP_dataset_denoised(torch.utils.data.Dataset):
	def __init__(self,args):
		self.args = args

		self.all_texts, self.all_megs = self.load_words_and_megs()

		self.uniq_words = self.get_uniq_words()
		self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
		self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}

		self.texts, self.megs = self.load_data()

	def load_words_and_megs(self, do_zscore = True):
		if self.args.chapter==1:
			mat = sio.loadmat(self.args.base_data_path+"meg/A_HP_notDetrended_25ms.mat")
			words = mat['labels']
			all_texts=[w[0][0].replace("@","").replace("\\","").replace("+","").replace("^","").strip() for w in words][:5176]
			
			all_megs=np.load(self.args.base_data_path+"denoised_HP.npy")
			all_megs=np.swapaxes(all_megs,1,2)
		elif self.args.chapter==2:
			words = np.load(self.args.base_data_path+"words_chapter_2.npy")
			all_texts=[w.replace("@","").replace("\\","").replace("+","").replace("^","").strip() for w in words]
			
			all_megs=np.load(self.args.base_data_path+"denoised_HP_chapter2.npy")
			all_megs=np.swapaxes(all_megs,1,2)

		if do_zscore:
			all_megs = zscore(all_megs)

		return all_texts, all_megs

	def load_data(self):
		texts_to_load=list()
		megs_to_load=list()
		
		megs=np.mean(self.all_megs[:,:,12:17],axis=2)
		for location in range(self.args.sequence_length,len(self.all_texts)):
			texts=self.all_texts[location-self.args.sequence_length+1:location+1]
			texts_to_load.append([self.word_to_index[w] for w in texts])
			megs_to_load.append(megs[location])
		
		texts_to_load=np.vstack(texts_to_load)
		megs_to_load=np.vstack(megs_to_load)
		assert len(texts_to_load)==len(megs_to_load)
		return texts_to_load, megs_to_load

	def get_uniq_words(self):
		words=set()
		for word in self.all_texts:
			words.add(word)
		return list(words)

	def __len__(self):
		return len(self.texts) 

	def __getitem__(self, index):
		# return word, meg
		return(
			torch.tensor(self.texts[index]),
			torch.tensor(self.megs[index+self.args.meg_offset])
		)