"""
Sampling script for DDPO-trained models.
Generates images from a checkpoint (or pretrained model) and saves them.
"""

import os
import datetime
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import ddpo_pytorch.prompts
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
import torch
from functools import partial
import tqdm
from PIL import Image
import contextlib

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")
flags.DEFINE_string("checkpoint_path", "", "Path to checkpoint directory (optional, if not provided uses pretrained model)")
flags.DEFINE_integer("num_samples", 100, "Number of images to generate")
flags.DEFINE_integer("batch_size", 8, "Batch size for generation")
flags.DEFINE_integer("num_inference_steps", None, "Number of inference steps (default: from config)")


def load_checkpoint(pipeline, checkpoint_path, config, device):
    """Load checkpoint (supports both LoRA and full UNet)."""
    checkpoint_path = os.path.normpath(os.path.expanduser(checkpoint_path))
    
    # If directory, find latest checkpoint
    if "checkpoint_" not in os.path.basename(checkpoint_path):
        checkpoints = list(
            filter(lambda x: "checkpoint_" in x, os.listdir(checkpoint_path))
        )
        if len(checkpoints) == 0:
            raise ValueError(f"No checkpoints found in {checkpoint_path}")
        checkpoint_path = os.path.join(
            checkpoint_path,
            sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
        )
    
    print(f"Loading checkpoint from {checkpoint_path}")
    
    if config.use_lora:
        # Load LoRA weights
        lora_dir = checkpoint_path
        lora_file = os.path.join(checkpoint_path, "pytorch_lora_weights.bin")
        if not os.path.exists(lora_file):
            raise FileNotFoundError(f"LoRA checkpoint not found at {lora_file}")
        pipeline.unet.load_attn_procs(lora_dir)
        print("LoRA weights loaded successfully!")
    else:
        # Load full UNet
        unet_path = os.path.join(checkpoint_path, "unet")
        if not os.path.exists(unet_path):
            raise FileNotFoundError(f"UNet checkpoint not found at {unet_path}")
        unet = UNet2DConditionModel.from_pretrained(unet_path)
        pipeline.unet.load_state_dict(unet.state_dict())
        del unet
        print("Full UNet weights loaded successfully!")


def main(_):
    config = FLAGS.config
    checkpoint_path = FLAGS.checkpoint_path
    num_samples = FLAGS.num_samples
    batch_size = FLAGS.batch_size
    num_inference_steps = FLAGS.num_inference_steps if FLAGS.num_inference_steps is not None else config.sample.num_steps
    
    # Setup accelerator
    accelerator = Accelerator()
    device = accelerator.device
    
    # Set seed
    set_seed(config.seed, device_specific=True)
    
    # Create output directory with unique run name
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    run_name = f"sample_{unique_id}"
    if checkpoint_path:
        checkpoint_name = os.path.basename(os.path.dirname(checkpoint_path) if "checkpoint_" in os.path.basename(checkpoint_path) else checkpoint_path)
        run_name = f"{checkpoint_name}_{unique_id}"
    
    output_dir = os.path.join("outputs", run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving samples to: {output_dir}")
    
    # Load pipeline
    print("Loading Stable Diffusion pipeline...")
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
    
    # Load checkpoint if provided
    if checkpoint_path:
        # Setup LoRA if needed (only set up LoRA processors if we have a checkpoint)
        if config.use_lora:
            # Set correct lora layers (will be loaded from checkpoint)
            lora_attn_procs = {}
            for name in pipeline.unet.attn_processors.keys():
                cross_attention_dim = (
                    None
                    if name.endswith("attn1.processor")
                    else pipeline.unet.config.cross_attention_dim
                )
                if name.startswith("mid_block"):
                    hidden_size = pipeline.unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[
                        block_id
                    ]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = pipeline.unet.config.block_out_channels[block_id]
                
                lora_attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
                )
            pipeline.unet.set_attn_processor(lora_attn_procs)
        
        load_checkpoint(pipeline, checkpoint_path, config, device)
    else:
        print("Using pretrained model (no checkpoint provided)")
        # If config says use_lora but no checkpoint, we use regular attention processors
        # (pipeline already has default processors, so we don't need to do anything)
        if config.use_lora:
            print("Warning: config.use_lora=True but no checkpoint provided. Using regular attention processors.")
    
    # Get prompt function
    prompt_fn = getattr(ddpo_pytorch.prompts, config.prompt_fn)
    
    # Pre-compute negative prompt embeddings
    negative_prompt = ""
    negative_prompt_ids = pipeline.tokenizer(
        negative_prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=pipeline.tokenizer.model_max_length,
    ).input_ids.to(device)
    negative_prompt_embeds = pipeline.text_encoder(negative_prompt_ids)[0]
    
    # Setup autocast
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    
    # Generate images
    pipeline.unet.eval()
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    all_prompts = []
    image_count = 0
    
    print(f"\nGenerating {num_samples} images...")
    for i in tqdm(range(num_batches), desc="Generating images"):
        current_batch_size = min(batch_size, num_samples - i * batch_size)
        
        # Generate prompts
        prompts, prompt_metadata = zip(
            *[
                prompt_fn(**config.prompt_fn_kwargs)
                for _ in range(current_batch_size)
            ]
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
        
        # Match negative prompt embeds shape to batch size
        neg_embeds = negative_prompt_embeds
        if neg_embeds.shape[0] != prompt_embeds.shape[0]:
            neg_embeds = neg_embeds.expand(prompt_embeds.shape[0], -1, -1)
        
        # Sample images
        with torch.no_grad(), autocast():
            images, _, _, _ = pipeline_with_logprob(
                pipeline,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=neg_embeds,
                num_inference_steps=num_inference_steps,
                guidance_scale=config.sample.guidance_scale,
                eta=config.sample.eta,
                output_type="pil",
            )
        
        # Save images
        for j, (image, prompt) in enumerate(zip(images, prompts)):
            image_path = os.path.join(output_dir, f"image_{image_count:06d}.png")
            image.save(image_path)
            
            # Save prompt to text file
            prompt_path = os.path.join(output_dir, f"image_{image_count:06d}_prompt.txt")
            with open(prompt_path, "w") as f:
                f.write(prompt)
            
            image_count += 1
        
        all_prompts.extend(prompts)
    
    # Save summary
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Sampling Summary\n")
        f.write(f"{'='*80}\n")
        f.write(f"Run name: {run_name}\n")
        f.write(f"Checkpoint: {checkpoint_path if checkpoint_path else 'Pretrained model'}\n")
        f.write(f"Number of samples: {num_samples}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Inference steps: {num_inference_steps}\n")
        f.write(f"Guidance scale: {config.sample.guidance_scale}\n")
        f.write(f"Eta: {config.sample.eta}\n")
        f.write(f"Prompt function: {config.prompt_fn}\n")
        f.write(f"Config: {config.pretrained.model}\n")
        f.write(f"{'='*80}\n")
        f.write(f"\nAll prompts:\n")
        for idx, prompt in enumerate(all_prompts):
            f.write(f"{idx:06d}: {prompt}\n")
    
    print(f"\n✅ Generated {image_count} images and saved to {output_dir}")
    print(f"   Summary saved to: {summary_path}")


if __name__ == "__main__":
    app.run(main)
