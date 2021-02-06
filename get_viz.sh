#!/bin/bash
#MODEL_NAMES="hde hde_no_types_no_pos"
MODEL_NAMES="hde"
#MODEL_NAMES="hde_no_types_no_pos"

for val in $MODEL_NAMES; do
    scp sacton@lengau.chpc.ac.za:/home/sacton/lustre/GNN_Thesis/Code/HDE/Checkpoint/${val}_losses.png Viz/${val}_losses.png
done