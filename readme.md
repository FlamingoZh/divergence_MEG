# Divergences between Language Models and Human Brains (Neurips 2024)

This repository provides implementations of analyses in the paper [Divergences between Language Models and Human Brains](https://arxiv.org/abs/2311.09308).



## Getting Started

### Clone the repository from GitHub

```
git clone git@github.com:FlamingoZh/divergence_MEG.git
```

### Requirements

- Python >= 3.10
- Other dependencies: numpy, scipy, scikit-learn, pandas, pytorch, torchvision, transformers, jupyterlab, matplotlib, seaborn

### Datasets

The word stimuli and preprocessed MEG recordings used in our study — including the Harry Potter and Moth datasets — can be accessed [here]((https://cmu.box.com/s/2sg5rfvfc4cl4yu8w6la23eb8ncrnv61)) via Box.

### LM Embeddings Generation

`scripts/gen_lm_embeddings.py` generates LM embeddings for every word in the corpus.

### Ridge Regression

`scripts/gen_data_for_analysis.py` runs ridge regression to predict human brain responses based on LM embeddings and dumps the MSE for each word's prediction.


### Hypothesis Proposing

1. Run `scripts/format_sentence.py` to select sentences with high and low MSEs and format them.
2. We adopted the automatic hypothesis proposer from Zhong et al. (2023). Please refer to their [GitHub repository](https://github.com/ruiqi-zhong/D5) for instructions on downloading the models and running the proposer and verifier.


### Fine-tuning

The file `finetuning/finetune_commonsense.py` provides code for fine-tuning models on two datasets: [social_i_qa](https://huggingface.co/datasets/allenai/social_i_qa) and [piqa](https://huggingface.co/datasets/ybisk/piqa).


### Human Experiments

1. We recruit subjects to verify the hypotheses proposed by the automatic method. The responses can be found in `human_experiments/hypothesis_verification`.
2. We provide word category annotations from three subjects for the Harry Potter datasets in `human_experiments/word_category_annot`.