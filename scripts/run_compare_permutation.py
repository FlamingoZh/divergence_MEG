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
	parser.add_argument("--chapter", type=int, default=1)

	parser.add_argument("--base_path", default=f"{home}/Desktop/Harry_divergence/interim_data/data_for_analysis/")
	parser.add_argument("--data_base_path")
	parser.add_argument("--data_ft_path")
	parser.add_argument("--ft_model_info")

	parser.add_argument("--n_sim", type=int, default=10000)

	parser.add_argument("--dump", action="store_false", help="dump generated data (by default True)")
	
	args = parser.parse_args()

	print("Echo arguments:",args)	

	## Load data

	data_base=pickle.load(open(f"{args.base_path}{args.data_base_path}","rb"))
	data_ft=pickle.load(open(f"{args.base_path}{args.data_ft_path}","rb"))


	## Run permutation
	pvals=utils.perm_test(data_base['actual_all_tw'],data_ft['pred_all_tw'],data_base['pred_all_tw'],args.n_sim)

	if args.dump:
		pickle.dump(pvals,
				open(f"../interim_data/permutation/{args.dataset}_chapter_{args.chapter}_base_and_{args.ft_model_info}_pvals.pkl","wb"))

