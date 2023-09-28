#!/bin/bash
#SBATCH --job-name=cv_base
#SBATCH --output=cv_base.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:4
#SBATCH --mem=50G
#SBATCH --time=10-00:00:00
#SBATCH --mail-user=emmy@cmu.edu
#SBATCH --mail-type=END
#SBATCH --exclude=tir-1-36,tir-1-11,tir-1-32

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mengyan3/miniconda3/lib/
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ds-gpt2-train

module load cuda-11.1.1
module load gcc-7.4

export HF_HOME=/scratch/mengyan3
#wandb connection issues
export WANDB_MODE=offline

deepspeed --num_gpus 4 finetune_commonsense.py --model gpt2-xl \
    --specific_dataset social_i_qa \
    --output_dir book4_cv/base \
    --num_epochs 3 \
    --wandb \
    --deepspeed \
    --lm_percentage 1 \
    --book4_cv \
    --wandb_offline
