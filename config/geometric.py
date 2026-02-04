"""
Configuration for training with geometric vanishing point reward.
Optimized for speed while maintaining good accuracy.
"""
import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    
    ###### General ######
    config.run_name = ""
    config.seed = 42
    config.logdir = "logs"
    config.num_checkpoint_limit = 5
    config.mixed_precision = "fp16"
    config.use_lora = True
    config.allow_tf32 = True
    
    ###### Pretrained Model ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    pretrained.model = "CompVis/stable-diffusion-v1-4"
    pretrained.revision = "main"
    
    ###### Sampling ######
    config.sample = sample = ml_collections.ConfigDict()
    sample.num_steps = 25  # Reduced from 50 for faster sampling
    sample.guidance_scale = 5.0
    sample.eta = 1.0
    sample.batch_size = 1
    sample.num_batches_per_epoch = 16
    
    ###### Training ######
    config.train = train = ml_collections.ConfigDict()
    train.batch_size = 1
    train.num_inner_epochs = 1
    train.use_8bit_adam = False
    train.learning_rate = 5e-4
    train.adam_beta1 = 0.9
    train.adam_beta2 = 0.999
    train.adam_weight_decay = 1e-4
    train.adam_epsilon = 1e-8
    train.gradient_accumulation_steps = 4
    train.max_grad_norm = 1.0
    train.num_train_timesteps = 1000
    train.cfg = True
    train.adv_clip_max = 5
    train.clip_advantages = True
    train.clip_range = 1e-4
    train.timestep_fraction = 1.0
    
    ###### Prompt and Reward Functions ######
    config.prompt_fn = "manhattan_scenes"
    config.prompt_fn_kwargs = {}
    
    # Use Algebraic Intersection reward (GPU-accelerated)
    config.reward_fn = "geometric_algebraic"
    config.reward_fn_kwargs = {
        "num_samples": 5,      
        "threshold_c": 0.03,
        "min_length": 15.0,
    }
    
    ###### Per-Prompt Stat Tracking ######
    config.per_prompt_stat_tracking = per_prompt_stat_tracking = ml_collections.ConfigDict()
    per_prompt_stat_tracking.buffer_size = 16
    per_prompt_stat_tracking.min_count = 16
    
    ###### Epochs and Logging ######
    config.num_epochs = 200
    config.save_freq = 20
    config.resume_from = ""
    
    return config
