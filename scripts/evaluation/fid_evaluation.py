"""
FID Evaluation Script for DDPO-trained models.
Adapted from pytorch-fid and denoising-diffusion-pytorch evaluation code.
"""

import math
import os
import subprocess
import sys
import glob
from absl import app, flags
from ml_collections import config_flags
import numpy as np
import torch
from torch.nn.functional import adaptive_avg_pool2d
from tqdm.auto import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Try to import pytorch-fid
try:
    from pytorch_fid.fid_score import calculate_frechet_distance
    from pytorch_fid.inception import InceptionV3
except ImportError:
    raise ImportError(
        "pytorch-fid is required for FID evaluation. Install it with: "
        "pip install pytorch-fid or git clone https://github.com/mseitzer/pytorch-fid.git && pip install -e ."
    )

# Try to import einops
try:
    from einops import rearrange, repeat
except ImportError:
    raise ImportError(
        "einops is required. Install it with: pip install einops"
    )

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob

import ddpo_pytorch.prompts

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")
flags.DEFINE_string("checkpoint_path", "logs/compressibility_30_2026.01.11_20.00.45/checkpoint_100", "Path to checkpoint directory (e.g., logs/run_name/checkpoint_100)")
flags.DEFINE_integer("num_fid_samples", 50000, "Number of samples to generate for FID calculation")
flags.DEFINE_integer("batch_size", 8, "Batch size for generation")
flags.DEFINE_string("real_images_dir", "", "Optional: Directory containing real images for FID calculation")
flags.DEFINE_string("generated_images_dir", "", "Optional: Directory containing generated images (if not provided, uses latest folder in outputs/)")
flags.DEFINE_string("stats_dir", "./fid_stats", "Directory to save/load dataset statistics")
flags.DEFINE_integer("inception_block_idx", 2048, "Inception block index for feature extraction")
flags.DEFINE_integer("num_inference_steps", None, "Override number of inference steps (default: from config, use 20-30 for faster generation)")


def num_to_groups(num, divisor):
    """Split num into groups of size divisor."""
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class ImageDataset(Dataset):
    """Dataset for loading real images from a directory."""
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        
        # Find all image files
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.JPEG', '*.PNG', '*.JPG']:
            self.image_paths.extend(
                [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                 if f.lower().endswith(ext.replace('*', ''))]
            )
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


class PipelineSampler:
    """Wrapper around Stable Diffusion pipeline to act as a sampler for FID evaluation."""
    def __init__(self, pipeline, prompt_fn, prompt_fn_kwargs, device, num_inference_steps=50, 
                 guidance_scale=5.0, eta=1.0):
        self.pipeline = pipeline
        self.prompt_fn = prompt_fn
        self.prompt_fn_kwargs = prompt_fn_kwargs
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.eta = eta
        
        # Pre-compute negative prompt embeddings
        negative_prompt = ""
        negative_prompt_ids = pipeline.tokenizer(
            negative_prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(device)
        self.negative_prompt_embeds = pipeline.text_encoder(negative_prompt_ids)[0]
    
    def sample(self, batch_size):
        """Generate a batch of images."""
        # Generate prompts
        prompts, _ = zip(*[
            self.prompt_fn(**self.prompt_fn_kwargs) 
            for _ in range(batch_size)
        ])
        
        # Encode prompts
        prompt_ids = self.pipeline.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.pipeline.tokenizer.model_max_length,
        ).input_ids.to(self.device)
        prompt_embeds = self.pipeline.text_encoder(prompt_ids)[0]
        
        # Generate images using the same sampling method as training
        with torch.no_grad():
            # Match negative prompt embeds shape to batch size
            neg_embeds = self.negative_prompt_embeds
            if neg_embeds.shape[0] != prompt_embeds.shape[0]:
                neg_embeds = neg_embeds.expand(prompt_embeds.shape[0], -1, -1)
            # Use pipeline_with_logprob for consistency with training (same as train.py)
            # output_type="pt" returns tensors in [0, 1] range, shape [B, C, H, W]
            images, _, _, _ = pipeline_with_logprob(
                self.pipeline,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=neg_embeds,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                eta=self.eta,
                output_type="pt",
            )
        
        # images is already a tensor in [0, 1] range with shape [B, C, H, W]
        # Ensure it's on the correct device and properly formatted
        if images.dim() == 4 and images.shape[1] == 3:
            # Already in [B, C, H, W] format
            return images.to(self.device)
        else:
            # Fallback: convert if needed
            return images.to(self.device)


class FIDEvaluation:
    def __init__(
        self,
        batch_size,
        sampler=None,
        generated_images_dir=None,
        accelerator=None,
        stats_dir="./fid_stats",
        device="cuda",
        num_fid_samples=50000,
        inception_block_idx=2048,
        real_images_dl=None,
    ):
        self.batch_size = batch_size
        self.n_samples = num_fid_samples
        self.device = device
        self.sampler = sampler
        self.generated_images_dir = generated_images_dir
        self.stats_dir = stats_dir
        self.real_images_dl = real_images_dl
        self.print_fn = print if accelerator is None else accelerator.print
        
        assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
        self.inception_v3 = InceptionV3([block_idx]).to(device)
        self.dataset_stats_loaded = False

    def calculate_inception_features(self, samples):
        """Calculate Inception features for a batch of images."""
        # Ensure images are in [0, 1] range and have 3 channels
        if samples.max() > 1.0:
            samples = samples / 255.0
        
        # Convert to float32 (Inception model expects float32, not float16)
        if samples.dtype != torch.float32:
            samples = samples.float()
        
        if samples.shape[1] == 1:
            samples = repeat(samples, "b 1 ... -> b c ...", c=3)
        elif samples.shape[1] != 3:
            raise ValueError(f"Expected 1 or 3 channels, got {samples.shape[1]}")

        self.inception_v3.eval()
        features = self.inception_v3(samples)[0]

        if features.size(2) != 1 or features.size(3) != 1:
            features = adaptive_avg_pool2d(features, output_size=(1, 1))
        features = rearrange(features, "... 1 1 -> ...")
        return features

    def load_or_precalc_dataset_stats(self):
        """Load or pre-calculate statistics for real images."""
        os.makedirs(self.stats_dir, exist_ok=True)
        path = os.path.join(self.stats_dir, "dataset_stats")
        
        if self.real_images_dl is None:
            self.print_fn("No real images provided. FID will be computed only on generated images.")
            self.dataset_stats_loaded = True
            return
        
        try:
            ckpt = np.load(path + ".npz")
            self.m2, self.s2 = ckpt["m2"], ckpt["s2"]
            self.print_fn("Dataset stats loaded from disk.")
            ckpt.close()
        except OSError:
            self.print_fn(
                f"Calculating Inception features for {self.n_samples} samples from the real dataset."
            )
            stacked_real_features = []
            
            num_batches = 0
            for real_samples in tqdm(self.real_images_dl, desc="Processing real images"):
                real_samples = real_samples.to(self.device)
                real_features = self.calculate_inception_features(real_samples)
                stacked_real_features.append(real_features)
                num_batches += 1
                
                # Limit to n_samples
                if len(stacked_real_features) * self.batch_size >= self.n_samples:
                    break
            
            stacked_real_features = torch.cat(stacked_real_features, dim=0).cpu().numpy()
            
            # Trim to exactly n_samples
            if len(stacked_real_features) > self.n_samples:
                stacked_real_features = stacked_real_features[:self.n_samples]
            
            m2 = np.mean(stacked_real_features, axis=0)
            s2 = np.cov(stacked_real_features, rowvar=False)
            np.savez_compressed(path, m2=m2, s2=s2)
            self.print_fn(f"Dataset stats cached to {path}.npz for future use.")
            self.m2, self.s2 = m2, s2
        
        self.dataset_stats_loaded = True

    @torch.inference_mode()
    def fid_score(self):
        """Calculate FID score."""
        if not self.dataset_stats_loaded:
            self.load_or_precalc_dataset_stats()
        
        stacked_fake_features = []
        
        # If we have a directory of generated images, load from there
        if self.generated_images_dir:
            self.print_fn(
                f"Loading generated images from {self.generated_images_dir}"
            )
            # Load images from directory (only image files, exclude prompt text files)
            image_paths = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.JPEG', '*.PNG', '*.JPG']:
                image_paths.extend(glob.glob(os.path.join(self.generated_images_dir, ext)))
            
            # Filter out prompt text files (files ending with _prompt.txt)
            image_paths = [p for p in image_paths if not p.endswith('_prompt.txt')]
            
            # Sort to ensure consistent ordering
            image_paths = sorted(image_paths)
            
            # Limit to n_samples
            if len(image_paths) > self.n_samples:
                image_paths = image_paths[:self.n_samples]
                self.print_fn(f"Using first {self.n_samples} images from {len(image_paths)} available")
            
            # Load and process images in batches
            transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ])
            
            self.print_fn(
                f"Calculating Inception features for {len(image_paths)} generated samples."
            )
            
            for i in tqdm(range(0, len(image_paths), self.batch_size), desc="Processing images"):
                batch_paths = image_paths[i:i + self.batch_size]
                # Load batch of images
                images = []
                for img_path in batch_paths:
                    img = Image.open(img_path).convert('RGB')
                    # Convert to tensor [C, H, W] in range [0, 1]
                    img_tensor = transform(img)
                    images.append(img_tensor)
                
                if images:
                    images_tensor = torch.stack(images).to(self.device)
                    fake_features = self.calculate_inception_features(images_tensor)
                    stacked_fake_features.append(fake_features)
        
        # Otherwise, use sampler (old method)
        elif self.sampler is not None:
            batches = num_to_groups(self.n_samples, self.batch_size)
            self.print_fn(
                f"Calculating Inception features for {self.n_samples} generated samples."
            )
            for batch in tqdm(batches, desc="Generating images"):
                fake_samples = self.sampler.sample(batch_size=batch)
                fake_features = self.calculate_inception_features(fake_samples)
                stacked_fake_features.append(fake_features)
        else:
            raise ValueError("Either sampler or generated_images_dir must be provided")
        
        stacked_fake_features = torch.cat(stacked_fake_features, dim=0).cpu().numpy()
        
        # Trim to exactly n_samples
        if len(stacked_fake_features) > self.n_samples:
            stacked_fake_features = stacked_fake_features[:self.n_samples]
        
        m1 = np.mean(stacked_fake_features, axis=0)
        s1 = np.cov(stacked_fake_features, rowvar=False)

        if self.real_images_dl is None:
            self.print_fn("No real images provided. Cannot compute FID.")
            return None
        
        fid = calculate_frechet_distance(m1, s1, self.m2, self.s2)
        return fid


def find_latest_output_folder(outputs_dir="outputs"):
    """Find the latest output folder in the outputs directory."""
    if not os.path.exists(outputs_dir):
        raise ValueError(f"Outputs directory {outputs_dir} does not exist")
    
    # Get all subdirectories
    subdirs = [d for d in os.listdir(outputs_dir) 
               if os.path.isdir(os.path.join(outputs_dir, d))]
    
    if not subdirs:
        raise ValueError(f"No output folders found in {outputs_dir}")
    
    # Sort by modification time (newest first)
    subdirs_with_time = [
        (d, os.path.getmtime(os.path.join(outputs_dir, d)))
        for d in subdirs
    ]
    subdirs_with_time.sort(key=lambda x: x[1], reverse=True)
    
    latest_folder = os.path.join(outputs_dir, subdirs_with_time[0][0])
    return latest_folder


def call_sample_script(config_path, checkpoint_path, num_samples, batch_size, num_inference_steps=None):
    """Call sample.py as a subprocess to generate images."""
    script_path = os.path.join(os.path.dirname(__file__), "..", "sample.py")
    script_path = os.path.normpath(script_path)
    
    cmd = [
        sys.executable,
        script_path,
        "--config", config_path,
        "--checkpoint_path", checkpoint_path,
        "--num_samples", str(num_samples),
        "--batch_size", str(batch_size),
    ]
    
    if num_inference_steps is not None:
        cmd.extend(["--num_inference_steps", str(num_inference_steps)])
    
    print(f"\n{'='*80}")
    print("Calling sample.py to generate images...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, check=True, capture_output=False)
    
    if result.returncode != 0:
        raise RuntimeError(f"sample.py failed with return code {result.returncode}")
    
    print(f"\n{'='*80}")
    print("Sample generation completed!")
    print(f"{'='*80}\n")


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
    num_fid_samples = FLAGS.num_fid_samples
    batch_size = FLAGS.batch_size
    real_images_dir = FLAGS.real_images_dir
    generated_images_dir = FLAGS.generated_images_dir
    stats_dir = FLAGS.stats_dir
    inception_block_idx = FLAGS.inception_block_idx
    
    if not checkpoint_path:
        raise ValueError("--checkpoint_path is required")
    
    # Get config path string from command line args
    # We need to find the --config argument from sys.argv
    config_path = "config/base.py"  # default
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            break
    
    # Override inference steps if provided
    num_inference_steps = FLAGS.num_inference_steps if FLAGS.num_inference_steps is not None else config.sample.num_steps
    
    # Step 1: Call sample.py to generate images
    # call_sample_script(
    #     config_path=config_path,
    #     checkpoint_path=checkpoint_path,
    #     num_samples=num_fid_samples,
    #     batch_size=batch_size,
    #     num_inference_steps=num_inference_steps,
    # )
    
    # Step 2: Determine the output folder for generated images
    if generated_images_dir:
        latest_output_folder = generated_images_dir
        print(f"Using specified generated images directory: {latest_output_folder}")
    else:
        # Find the latest output folder if not specified
        try:
            latest_output_folder = find_latest_output_folder()
            print(f"Using latest generated images folder: {latest_output_folder}")
        except (ValueError, OSError) as e:
            raise ValueError(
                f"Could not find generated images directory. Please specify --generated_images_dir "
                f"or ensure outputs/ directory exists with generated images. Error: {e}"
            )
    
    # Setup accelerator
    accelerator = Accelerator()
    device = accelerator.device
    
    # Setup real images dataloader if provided
    real_images_dl = None
    if real_images_dir:
        print(f"Loading real images from {real_images_dir}")
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        dataset = ImageDataset(real_images_dir, transform=transform)
        real_images_dl = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
    
    # Create FID evaluator (using generated_images_dir instead of sampler)
    fid_evaluator = FIDEvaluation(
        batch_size=batch_size,
        sampler=None,
        generated_images_dir=latest_output_folder,
        accelerator=accelerator,
        stats_dir=stats_dir,
        device=device,
        num_fid_samples=num_fid_samples,
        inception_block_idx=inception_block_idx,
        real_images_dl=real_images_dl,
    )
    
    # Calculate FID
    print("\n" + "="*80)
    print("Starting FID Evaluation")
    print("="*80)
    fid_score = fid_evaluator.fid_score()
    
    if fid_score is not None:
        print("\n" + "="*80)
        print(f"FID Score: {fid_score:.4f}")
        print(f"Generated images from: {latest_output_folder}")
        print("="*80)
    else:
        print("\nFID score could not be computed (no real images provided).")


if __name__ == "__main__":
    app.run(main)

