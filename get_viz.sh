#!/bin/bash
MODEL_NAMES="stack_large_deep stack_large"


for val in $MODEL_NAMES; do
    scp sacton@lengau.chpc.ac.za:/home/sacton/lustre/GNN_Thesis/Code/HDE/Checkpoint/hde_model_${val}_losses.png Viz/hde_model_${val}_losses.png
done