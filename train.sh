#!/bin/bash
EMAIL="$1"
LOGFILE="./logs/training_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOGFILE")"
{
    echo "Training started at $(date)"
    echo "----------------------------------------"
    
    python3 train_model_nohierarchy.py \
      --batch_size 24        \
      --batch_size_test 24   \
      --max_batches 1000000  \
      --save_every 5         \
      --test_every 1000      \
      --base_model clip      \
      --clip_model_name openai/clip-vit-large-patch14 \
      --depth 12             \
      --embed_dim 2048       \
      --num_hidden_dims 8192 \
      --heads 32             \
      --compile              \
#     --freeze_clip \
      2>&1
    
    echo "----------------------------------------"
    echo "Training finished at $(date)"
} | tee "$LOGFILE"

# Capture the exit code
EXIT_CODE=${PIPESTATUS[0]}

# Construct email content
if [[ $EXIT_CODE -eq 0 ]]; then
EMAIL_CONTENT=$(cat << EOF
Subject: Training Completed Successfully - $(date)

Training completed successfully at $(date)
================================
$(tail -n 100 "$LOGFILE")
EOF
)

else
EMAIL_CONTENT=$(cat << EOF
Subject: Training Failed (Exit Code: $EXIT_CODE) - $(date)

Training failed with exit code $EXIT_CODE at $(date)
================================
$(tail -n 100 "$LOGFILE")
EOF
)
fi

# Email the results if email provided and mail command exists
if [[ -n "$EMAIL" ]] && command -v sendmail >/dev/null 2>&1; then
    echo "$EMAIL_CONTENT" | sendmail "$EMAIL"
elif [[ -n "$EMAIL" ]] && command -v msmtp >/dev/null 2>&1; then
    echo "$EMAIL_CONTENT" | msmtp "$EMAIL"
fi