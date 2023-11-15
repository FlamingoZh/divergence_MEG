#!/bin/bash
#SBATCH --job-name=fig_model
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:4
#SBATCH --mem=70G
#SBATCH --time=10-00:00:00

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mengyan3/miniconda3/lib/
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ds-gpt2-train

module load cuda-11.1.1
module load gcc-7.4

export HF_HOME=/scratch/mengyan3

# MCQ
deepspeed --num_gpus 4 finetune_commonsense.py --model gpt2-xl \
    --specific_dataset nightingal3/fig-qa \
    --output_dir finetuned_models \
    --early_stopping 3 \
    --wandb \
    --deepspeed \
    --lm_percentage 0
