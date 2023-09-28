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

def color_texts(texts,mses,offset,colors,write_path,filename):
	assert len(texts)==len(mses)+offset
	
	# get rank of each element
	sorted_idx = np.argsort(mses)
	ranks = np.empty_like(sorted_idx)
	ranks[sorted_idx] = np.arange(len(mses))
	
	with open(f"{write_path}{filename}","w", encoding='utf-8') as f:
		# f.write(".box {float: left; height: 20px; width: 20px; margin-bottom: 15px; border: 1px solid black; clear: both;}")
		for color in colors:
			f.write(f"<div style='display: inline-block; height: 30px; width: 30px; background-color: {color};'></div>")
		f.write("<div>From most different to least different.</div>")
		
		for i,word in enumerate(texts):
			if i<=offset:
				# print(i)
				f.write(f"<span style='color:{colors[-1]};'>"+word+" </span>")
			else:
				color=colors[int(ranks[i-offset]/len(ranks)*len(colors))]
				# print(color)
				f.write(f"<span style='color:{color};'>"+word+" </span>")

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	parser.add_argument("--dataset", default="HP")
	parser.add_argument("--chapter", type=int, default=1)
	parser.add_argument("--offset", type=int, default=100)

	parser.add_argument("--base_path", default="/home/yuchen/Desktop/Harry_divergence/interim_data/data_for_analysis/")
	parser.add_argument("--data_path", default="HP_chapter_1_base.pkl")

	parser.add_argument("--write_path", default="/home/yuchen/Desktop/Harry_divergence/interim_data/colored_text/")
	parser.add_argument("--model_info", default="base")

	args = parser.parse_args()

	## Load data

	data=pickle.load(open(f"{args.base_path}{args.data_path}","rb"))

	texts=data['text']
	mses=np.mean(data['MSE_all_tw'][7:12],axis=0)

	# colors=["#ff1818", "#d20000", "#a30002", "#720001", "#670061", "#5e1a8b", "#3d0064", "#555555", "#343434", "#000000"]
	colors=["#d20000", "#720001", "#5e1a8b", "#555555", "#000000"]

	filename=f"{args.model_info}_{args.dataset}_chapter_{args.chapter}.html"
	color_texts(texts,mses,args.offset,colors,args.write_path,filename)

