#!/bin/bash

python3 train_model_nohierarchy.py \
  --batch_size 24 \
  --batch_size_test 24 \
  --max_batches 100000 \
  --save_every 1000 \
  --base_model clip \
  --clip_model_name openai/clip-vit-large-patch14 \
  --depth 12 \
  --embed_dim 2048 \
  --num_hidden_dims 8192 \
  --heads 32 \