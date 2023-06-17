#!/bin/bash

#SBATCH --job-name=oa
#SBATCH --partition=a100-cu117
#SBATCH --nodes=8
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=sft.txt
#SBATCH --error=sft.err
#SBATCH --exclusive
#SBATCH --account=stablegpt
#SBATCH --mem=0


source /mnt/nvme/home/duyphung/.bashrc

conda env list
eval "$(conda shell.bash hook)"
conda activate ax_env

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901
export WANDB_ENTITY=pvduy

cd /mnt/nvme/home/duyphung/Custom-Open-Assistant/model/model_training

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    trainer_sft.py --configs defaults openllama-instruct wizard-only --cache_dir data_cache --deepspeed'
