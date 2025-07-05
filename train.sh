#!/bin/bash

EMAIL="$1"
LOGFILE="./logs/training_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$(dirname "$LOGFILE")"

{
    echo "Training started at $(date)"
    echo "----------------------------------------"
    
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
      --compile \
      2>&1
    
    echo "----------------------------------------"
    echo "Training finished at $(date)"
} | tee "$LOGFILE"

# Email the results if sendmail exists and email provided
if [[ -n "$EMAIL" ]] && command -v sendmail >/dev/null 2>&1; then
    echo "Training completed - $(date)" | sendmail "$EMAIL" < "$LOGFILE"
elif [[ -n "$EMAIL" ]] && command -v msmtp >/dev/null 2>&1; then
    echo "Training completed - $(date)" | msmtp "$EMAIL" < "$LOGFILE"
fi