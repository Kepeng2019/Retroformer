#!/bin/bash

checkpoints=("model_800000.pt")
for checkpoint in "${checkpoints[@]}"; do
    echo "Running translation for checkpoint: $checkpoint"
    start_time=$(date +%s)
    python retroformer/translate.py \
        --batch_size_val 1 \
        --shared_vocab "False" \
        --shared_encoder False \
        --data_dir './data' \
        --intermediate_dir './intermediate' \
        --checkpoint_dir './checkpoint' \
        --checkpoint "$checkpoint" \
        --known_class 'True' \
        --beam_size 3 \
        --stepwise "False" \
        --use_template False \
        --mode "test_top10"
    end_time=$(date +%s)
    runtime=$((end_time - start_time))
    hours=$(echo "scale=2; $runtime/3600" | bc)
    echo "Checkpoint $checkpoint run time: $hours hours"
    echo "----------------------------------------"
done