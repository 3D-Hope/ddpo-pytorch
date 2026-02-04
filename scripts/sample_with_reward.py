#!/usr/bin/env python3
"""
Sample images and calculate geometric rewards.
Usage: python scripts/sample_with_reward.py --config=config/geometric.py --num_samples=100
"""

import os
import datetime
import numpy as np
import json
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import StableDiffusionPipeline, DDIMScheduler
import ddpo_pytorch.prompts
import ddpo_pytorch.rewards
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
import torch
from functools import partial
import tqdm
import contextlib

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Configuration file.")
flags.DEFINE_integer("num_samples", 100, "Number of images to generate")
flags.DEFINE_integer("batch_size", 4, "Batch size for generation")


def main(_):
    config = FLAGS.config
    num_samples = FLAGS.num_samples
    batch_size = FLAGS.batch_size
    
    # Setup
    accelerator = Accelerator()
    device = accelerator.device
    set_seed(config.seed, device_specific=True)
    
    # Create output directory
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    run_name = f"sample_reward_{config.reward_fn}_{unique_id}"
    output_dir = os.path.join("outputs", run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print(f"Sampling with Geometric Reward")
    print("=" * 80)
    print(f"Reward function: {config.reward_fn}")
    print(f"Prompt function: {config.prompt_fn}")
    print(f"Number of samples: {num_samples}")
    print(f"Batch size: {batch_size}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Load pipeline
    print("\nLoading Stable Diffusion pipeline...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model, revision=config.pretrained.revision
    )
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)
    pipeline.safety_checker = None
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    
    # Move to device
    inference_dtype = torch.float32
    if config.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16
    
    pipeline.vae.to(device, dtype=inference_dtype)
    pipeline.text_encoder.to(device, dtype=inference_dtype)
    pipeline.unet.to(device, dtype=inference_dtype)
    
    # Get prompt and reward functions
    prompt_fn = getattr(ddpo_pytorch.prompts, config.prompt_fn)
    reward_fn_kwargs = config.reward_fn_kwargs if hasattr(config, "reward_fn_kwargs") else {}
    reward_fn = getattr(ddpo_pytorch.rewards, config.reward_fn)(**reward_fn_kwargs)
    
    print(f"✓ Loaded reward function: {config.reward_fn}")
    print(f"✓ Loaded prompt function: {config.prompt_fn}")
    
    # Pre-compute negative prompt embeddings
    negative_prompt_ids = pipeline.tokenizer(
        "",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=pipeline.tokenizer.model_max_length,
    ).input_ids.to(device)
    negative_prompt_embeds = pipeline.text_encoder(negative_prompt_ids)[0]
    
    # Setup autocast
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    
    # ========== PHASE 1: GENERATE ALL IMAGES ==========
    pipeline.unet.eval()
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    all_images = []
    all_prompts = []
    all_metadata = []
    
    print(f"\n[PHASE 1/2] Generating {num_samples} images...")
    
    for i in tqdm(range(num_batches), desc="Generating"):
        current_batch_size = min(batch_size, num_samples - i * batch_size)
        
        # Generate prompts
        prompts, prompt_metadata = zip(
            *[prompt_fn(**config.prompt_fn_kwargs) for _ in range(current_batch_size)]
        )
        
        # Encode prompts
        prompt_ids = pipeline.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(device)
        prompt_embeds = pipeline.text_encoder(prompt_ids)[0]
        
        # Match negative prompt embeds
        neg_embeds = negative_prompt_embeds
        if neg_embeds.shape[0] != prompt_embeds.shape[0]:
            neg_embeds = neg_embeds.expand(prompt_embeds.shape[0], -1, -1)
        
        # Sample images
        with torch.no_grad(), autocast():
            images, _, _, _ = pipeline_with_logprob(
                pipeline,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=neg_embeds,
                num_inference_steps=config.sample.num_steps,
                guidance_scale=config.sample.guidance_scale,
                eta=config.sample.eta,
                output_type="pil",
            )
        
        # Store images and prompts
        all_images.extend(images)
        all_prompts.extend(prompts)
        all_metadata.extend(prompt_metadata)
    
    print(f"✓ Generated {len(all_images)} images")
    
    # ========== PHASE 2: CALCULATE REWARDS ==========
    print(f"\n[PHASE 2/2] Calculating rewards for {len(all_images)} images...")
    import time
    
    total_start = time.time()
    all_rewards = []
    
    # Calculate rewards sequentially to time each image
    for i, (image, prompt, metadata) in enumerate(zip(all_images, all_prompts, all_metadata)):
        start_t = time.time()
        
        # Convert single image to numpy batch for function
        img_np = np.array(image)[None]  # Add batch dim
        reward_batch, _ = reward_fn(img_np, [prompt], [metadata])
        reward = float(reward_batch[0])
        
        elapsed = time.time() - start_t
        all_rewards.append(reward)
        
        # Log every image
        print(f"Image {i+1}/{len(all_images)}: Reward={reward:.1f}, Time={elapsed:.2f}s")
        
    total_time = time.time() - total_start
    avg_time = total_time / len(all_images)
    
    print(f"\n✓ Calculated {len(all_rewards)} rewards in {total_time:.1f}s")
    print(f"  Average time per image: {avg_time:.2f}s")
    
    # ========== SAVE RESULTS ==========
    print(f"\nSaving images and results...")
    
    for i, (image, prompt, reward) in enumerate(zip(all_images, all_prompts, all_rewards)):
        # Save image
        img_path = os.path.join(output_dir, f"image_{i:06d}.png")
        image.save(img_path)
        
        # Save prompt and reward
        txt_path = os.path.join(output_dir, f"image_{i:06d}.txt")
        with open(txt_path, "w") as f:
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Reward: {reward:.4f}\n")
    
    # Compute statistics
    rewards_array = np.array(all_rewards)
    stats = {
        "reward_function": config.reward_fn,
        "prompt_function": config.prompt_fn,
        "num_samples": len(all_rewards),
        "rewards": {
            "mean": float(np.mean(rewards_array)),
            "std": float(np.std(rewards_array)),
            "min": float(np.min(rewards_array)),
            "max": float(np.max(rewards_array)),
            "median": float(np.median(rewards_array)),
        },
        "all_rewards": [float(r) for r in all_rewards],
        "all_prompts": all_prompts,
    }
    
    # Save results
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Generated {len(all_images)} images")
    print(f"\nReward Statistics:")
    print(f"  Mean:   {stats['rewards']['mean']:.4f}")
    print(f"  Std:    {stats['rewards']['std']:.4f}")
    print(f"  Min:    {stats['rewards']['min']:.4f}")
    print(f"  Max:    {stats['rewards']['max']:.4f}")
    print(f"  Median: {stats['rewards']['median']:.4f}")
    print("=" * 80)
    print(f"\n✓ Images saved to: {output_dir}")
    print(f"✓ Results saved to: {results_path}")


if __name__ == "__main__":
    app.run(main)
