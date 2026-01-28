# !/bin/bash

# conda activate <env name>

# Submit a SLURM job requesting 2 nodes for 1700 minutes
# Each node has 1 GPU (A100 or H100) and 20 CPUs
python submit.py \
    compute/greene=2x1 compute/greene/node=ah \
    compute.timeout=1700 \
    compute.cpus_per_task=20 \
    exp=midway_bdd \
    name='midway-bdd-slurm'
