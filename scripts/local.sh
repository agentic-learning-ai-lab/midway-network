# !/bin/bash

# conda activate <env name>

# Change number of nproc-per-node to equal number of GPUs
torchrun --standalone --nnodes=1 --nproc-per-node=1 submit.py \
    compute=local \
    exp=midway_wt_venice \
    name='midway-wt-venice-local' \
    wandb=false num_workers=4
