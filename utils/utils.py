import numpy as np
import sys
import os
import pickle
import matplotlib.pyplot as plt

from scipy.special import softmax
from scipy.stats import pearsonr
from scipy.stats import chi2

from sklearn.model_selection import KFold

import statsmodels.stats.multitest as smm

import torch

import mne
from mne.io import read_raw_ctf

from utils.ridge.ridge import bootstrap_ridge

def correlation_loss_wordwise(x,y):
	return correlation_loss(x,y,'wordwise')

def correlation_loss_channelwise(x,y):
	return correlation_loss(x,y,'channelwise')

def correlation_loss(x,y,mode):
	corr_list=correlation(x,y,mode)
	return -torch.mean(corr_list) # because we want to minimize loss/maximize correlation

def correlation(x,y,mode="channelwise"):
	valide_modes={'channelwise','wordwise'}
	if mode not in valide_modes:
		raise ValueError("Unknown Correlation Loss Function.")

	if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
		x=torch.tensor(x)
		y=torch.tensor(y)
	
	# if channelwise correlation
	xx=x.reshape(-1,x.shape[-1])
	yy=y.reshape(-1,y.shape[-1])
	# if wordwise correlation
	if mode=="wordwise":
		xx=xx.transpose(0, 1)
		yy=yy.transpose(0, 1)
		
	# print(xx.shape,yy.shape)
	# print(torch.mean(xx,axis=0).shape,torch.mean(yy,axis=0).shape)
	vx = xx - torch.mean(xx,axis=0)
	vy = yy - torch.mean(yy,axis=0)
	# print(torch.sum(vx * vy, dim=0).shape)
	cor_list=torch.sum(vx * vy, dim=0) * torch.rsqrt(torch.sum(vx ** 2,dim=0)) * torch.rsqrt(torch.sum(vy ** 2,dim=0))
	# print(cor_list.shape, cor_list)
	return cor_list

def compute_brain_surprisal_according_to_polarity(brain_data,neg_sig_channels=None,pos_sig_channels=None):
	# brain_data: N_data_point x N_channel
	if neg_sig_channels and pos_sig_channels:
		temp1=-brain_data[:,neg_sig_channels]
		temp2=brain_data[:,pos_sig_channels]
		temp=np.hstack([temp1,temp2])
		return np.mean(temp,axis=1)
	else:
		return np.mean(brain_data,axis=1)

def load_tokenizer_and_model_from_transformers(name):
	if name=="GPT-J":
		from transformers import AutoTokenizer, GPTJModel
		tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B",truncation_side="left",padding_side="left")
		model = GPTJModel.from_pretrained("EleutherAI/gpt-j-6B")
	elif name=="demo_GPT":
		from transformers import AutoTokenizer, GPTJModel
		tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gptj",truncation_side="left",padding_side="left")
		model = GPTJModel.from_pretrained("hf-internal-testing/tiny-random-gptj")
	elif name=="GPT2":
		from transformers import GPT2Tokenizer, GPT2Model
		tokenizer = GPT2Tokenizer.from_pretrained("gpt2",truncation_side="left",padding_side="left")
		model = GPT2Model.from_pretrained("gpt2")
	elif name=="GPT2-xl":
		from transformers import GPT2Tokenizer, GPT2Model
		tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl",truncation_side="left",padding_side="left")
		model = GPT2Model.from_pretrained("gpt2-xl")
	elif name=="GPT-Neo":
		from transformers import AutoTokenizer, GPTNeoModel
		tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B",truncation_side="left",padding_side="left")
		model = GPTNeoModel.from_pretrained("EleutherAI/gpt-neo-2.7B")
	elif name=="GPT-J_LMHead":
		from transformers import AutoTokenizer, GPTJForCausalLM
		tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B",truncation_side="left",padding_side="left")
		model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B",output_hidden_states=True)
	elif name=="demo_GPT_LMHead":
		from transformers import AutoTokenizer, GPTJForCausalLM
		tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gptj",truncation_side="left",padding_side="left")
		model = GPTJForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gptj",output_hidden_states=True)	
	elif name=="GPT2_LMHead":
		from transformers import AutoTokenizer, GPT2LMHeadModel
		tokenizer = AutoTokenizer.from_pretrained("gpt2",truncation_side="left",padding_side="left")
		model = GPT2LMHeadModel.from_pretrained("gpt2",output_hidden_states=True)
	elif name=="GPT2-xl_LMHead":
		from transformers import AutoTokenizer, GPT2LMHeadModel
		tokenizer = AutoTokenizer.from_pretrained("gpt2-xl",truncation_side="left",padding_side="left")
		model = GPT2LMHeadModel.from_pretrained("gpt2-xl",output_hidden_states=True)
	elif name=="GPT-Neo_LMHead":
		from transformers import AutoTokenizer, GPTNeoForCausalLM
		tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B",truncation_side="left",padding_side="left")
		model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B",output_hidden_states=True)	
	else:
		raise ValueError("Unknown model name.")
	return tokenizer, model

def find_indices_of_last_word(tokenizer,last_word,encodings):
	for word_end in range(len(encodings)-1,-1,-1):
		for word_start in range(word_end,-1,-1):
			reconstructed="".join([tokenizer.decode(i) for i in encodings[word_start:word_end+1]]).strip()
			if reconstructed==last_word:
				return (word_start, word_end)
	return (len(encodings)-1,len(encodings)-1)

def find_indices_of_last_word_in_batch(tokenizer,batch_last_words,batch_encodings):
	assert len(batch_last_words)==len(batch_encodings)
	return [find_indices_of_last_word(tokenizer,last_word,encodings) for last_word,encodings in zip(batch_last_words,batch_encodings)]

def compute_surprisal(all_last_words,all_logits):    
	assert len(all_logits)==len(all_last_words)
	all_surprisal=list()
	for token_ids, logits in zip(all_last_words,all_logits):
		assert len(token_ids)==len(logits)
		surprisal=0
		for token_id,logit in zip(token_ids,logits):
			p=softmax(logit)
			surprisal+=-1*np.log(p[token_id])
		all_surprisal.append(surprisal)
	return all_surprisal

def ceiling_division(n, d):
	return -(n // -d)

###############################
def Fisher_method(all_pvalues):
	chi=list()
	chi_pvalues=list()
	for pvalues in all_pvalues.T:
		chi_squared=-2*np.sum([np.log(p) for p in pvalues])
		chi_p=chi2.sf(chi_squared,2*len(pvalues))
		chi.append(chi_squared)
		chi_pvalues.append(chi_p)
	return np.array(chi),np.array(chi_pvalues)

def do_ridge_regression_and_compute_correlation(features,brain_responses,n_splits=10,p_thresh=0.001):
	kf = KFold(n_splits=n_splits)

	wts=list()
	all_y_test=list()
	all_y_test_pred=list()
	for train, test in kf.split(features):
		X_train, X_test, y_train, y_test = features[train], features[test], brain_responses[train], brain_responses[test]

		alphas = np.logspace(-1, 3, 10) # Equally log-spaced alphas between 10 and 1000. The third number is the number of alphas to test.
		nboots = 5 # Number of cross-validation runs.
		chunklen = 40 # 
		nchunks = 20

		wt, corr, alphas, bscorrs, valinds = bootstrap_ridge(X_train, y_train, X_test, y_test,
															alphas, nboots, chunklen, nchunks,
															singcutoff=1e-10, single_alpha=True)
		y_test_pred = np.dot(X_test, wt)
		
		all_y_test.append(y_test)
		all_y_test_pred.append(y_test_pred)

		wts.append(wt)

	all_y_test=np.vstack(all_y_test)
	all_y_test_pred=np.vstack(all_y_test_pred)

	all_cors=list()
	all_pvalues=list()
	for i,j in zip(y_test_pred.T,y_test.T):
		cor,pvalue=pearsonr(i,j,alternative="greater")
		all_cors.append(cor)
		all_pvalues.append(pvalue)

	wts=np.mean(np.stack(wts,axis=0),axis=0)

	is_pvalue_sig=[1 if pvalue<p_thresh else 0 for pvalue in all_pvalues]
	sig_channels=[i for i,p in enumerate(all_pvalues) if p<p_thresh]

	return wts,all_cors,all_pvalues,is_pvalue_sig,sig_channels


def cosine_similarity(A,B):
	return np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))


def perm_test(D,P1,P2,n=1000):
	mat=np.zeros((D.shape[2],D.shape[0]))
	for ch in range(D.shape[2]):
		print(ch,"...")
		for t in range(D.shape[0]):
			diff_true=pearsonr(D[t,:,ch], P1[t,:,ch])[0] - pearsonr(D[t,:,ch], P2[t,:,ch])[0]
			diff_list=list()
			for i in range(n):
				D_i=np.random.permutation(D[t,:,ch])
				diff_i=pearsonr(D_i, P1[t,:,ch])[0] - pearsonr(D_i, P2[t,:,ch])[0]
				diff_list.append(diff_i)
			p_value=(np.sum([item>diff_true for item in diff_list])+1)/(len(diff_list)+1)
			mat[ch,t]=p_value
	return mat

def FDR_correction(pvals):
	pvals=np.array(pvals)
	if len(pvals.shape)>1:
		pvals_reshape=np.reshape(pvals,np.product(pvals.shape))
	pvals_corrected=smm.fdrcorrection(pvals_reshape)[1]
	pvals_corrected=np.reshape(pvals_corrected,pvals.shape)
	return pvals_corrected
###############################

def plot_meg_to_cortex(sensor_file, corrs, fig, ax, file=False, trim_sensor=True):
	"""
	sensor_file: a file that has all information about MEG
	corr_file: (path to a file containing) correlations of channels
	"""

	raw_fname = os.path.join(sensor_file)
	raw = read_raw_ctf(raw_fname,verbose=False)

	# drop suffix
	if trim_sensor:
		new_channel_names=dict()
		for name in raw.info['ch_names']:
			new_name=name.split("-")[0]
			new_channel_names[name]=new_name
		
		raw.rename_channels(new_channel_names,verbose=False)
	
	temp=mne.channel_indices_by_type(raw.info)
	mag_info=mne.pick_info(raw.info,temp['mag'])

	if file:
		corrs=pickle.load(open(corrs,"rb"))
	# print(np.mean(corrs))

	# im,cm=mne.viz.plot_topomap(corrs,mag_info,show=False,sphere=0.2,axes=ax)
	im,cm=mne.viz.plot_topomap(corrs,mag_info,show=False,axes=ax)

	# manually fiddle the position of colorbar
	ax_x_start = 0.95
	ax_x_width = 0.04
	ax_y_start = 0.1
	ax_y_height = 0.9
	cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
	clb = fig.colorbar(im, cax=cbar_ax)

def plot_meg_to_cortex_multiple(sensor_file, corrs_list, fig, axes, file=False, trim_sensor=True):
	"""
	sensor_file: a file that has all information about MEG
	corr_file: (path to a file containing) correlations of channels
	"""

	raw_fname = os.path.join(sensor_file)
	raw = read_raw_ctf(raw_fname,verbose=False)

	# drop suffix
	if trim_sensor:
		new_channel_names=dict()
		for name in raw.info['ch_names']:
			new_name=name.split("-")[0]
			new_channel_names[name]=new_name
		
		raw.rename_channels(new_channel_names,verbose=False)
	
	temp=mne.channel_indices_by_type(raw.info)
	mag_info=mne.pick_info(raw.info,temp['mag'])

	if file:
		corrs=pickle.load(open(corrs,"rb"))
	# print(np.mean(corrs))

	assert len(corrs_list)==axes

	for i,ax in enumerate(axes):
		im,cm=mne.viz.plot_topomap(corrs,mag_info,show=False,axes=axes)

	# manually fiddle the position of colorbar
	ax_x_start = 0.95
	ax_x_width = 0.04
	ax_y_start = 0.1
	ax_y_height = 0.9
	cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
	clb = fig.colorbar(im, cax=cbar_ax)


# def load_glove(file_path):
# 	"""
# 	file_path: a txt file that stores glove embeddings
	
# 	Returns
# 	vocab_npa: all words
# 	embs_npa: a matrix that contains embeddings of words in the same order as vocab_npa
# 	"""
# 	vocab,embeddings = [],[]
	
# 	with open(file_path,'rt') as fi:
# 		full_content = fi.read().strip().split('\n')
	
# 	for i in range(len(full_content)):
# 		i_word = full_content[i].split(' ')[0]
# 		i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
# 		vocab.append(i_word)
# 		embeddings.append(i_embeddings)
	
# 	vocab_npa = np.array(vocab)
# 	embs_npa = np.array(embeddings)

# 	#insert '<pad>' and '<unk>' tokens at start of vocab_npa.
# 	vocab_npa = np.insert(vocab_npa, 0, '<pad>')
# 	vocab_npa = np.insert(vocab_npa, 1, '<unk>')
# 	#print(vocab_npa[:10])

# 	pad_emb_npa = np.zeros((1,embs_npa.shape[1]))   #embedding for '<pad>' token.
# 	unk_emb_npa = np.mean(embs_npa,axis=0,keepdims=True)    #embedding for '<unk>' token.

# 	#insert embeddings for pad and unk tokens at top of embs_npa.
# 	embs_npa = np.vstack((pad_emb_npa,unk_emb_npa,embs_npa))
# 	#print(embs_npa.shape)

# 	return vocab_npa, embs_npa