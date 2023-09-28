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

import pprint


if __name__ == '__main__':

	result=pickle.load(open("/home/yuchen/Desktop/Harry_divergence/interim_data/divergent_sents/hypotheses_both_chapters_arch.pkl","rb"))
	hyps=list(result)
	scores=[result[hyp]['diff_w_significance']['mu'] for hyp in hyps]
	p_values=[result[hyp]['diff_w_significance']['p_value'] for hyp in hyps]
	pprint.pprint([(hyps[idx],np.round(scores[idx],4),np.round(p_values[idx],4)) for idx in np.argsort(scores)[::-1]][:10])
