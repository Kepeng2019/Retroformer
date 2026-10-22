#!/bin/bash

CUDA_LAUNCH_BLOCKING=1 python ./retroformer/train.py \
  --encoder_num_layers 8 \
  --decoder_num_layers 8 \
  --heads 8 \
  --max_step 400000 \
  --batch_size_trn 1 \
  --batch_size_val 1 \
  --batch_size_token 4096 \
  --save_per_step 25000 \
  --val_per_step 25000 \
  --report_per_step 5000 \
  --device 'cuda' \
  --known_class 'False' \
  --shared_vocab "True" \
  --shared_encoder False \
  --data_dir './data' \
  --intermediate_dir './intermediate' \
  --checkpoint_dir './checkpoint'


