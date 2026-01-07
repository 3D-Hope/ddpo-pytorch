accelerate launch scripts/train_incremental.py \
    --config=config/incremental.py:incompressibility \
    --config.incremental_training=True
