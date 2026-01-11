accelerate launch scripts/train_incremental.py \
    --config=config/test_incremental.py:compressibility \
    --config.run_name=test_incremental_compressibility \
    --config.incremental_training=True \
    --config.incremental_min_steps=15


srun --partition=debug --qos=debug --gres=gpu:a6000:1 --time=01:00:00 --pty bash


accelerate launch \
    scripts/train_incremental_debug.py \
    --config=config/test_incremental.py:compressibility \
    --config.incremental_training=True \
    --config.incremental_min_steps=5 \
    --config.incremental_max_steps=50 \
    --config.incremental_constant_epochs=5