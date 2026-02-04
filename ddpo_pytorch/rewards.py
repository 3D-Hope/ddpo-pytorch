from PIL import Image
import io
import numpy as np
import torch


from PIL import Image
import io

def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew, meta

    return _fn


def aesthetic_score():
    from ddpo_pytorch.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


def llava_strict_satisfaction():
    """Submits images to LLaVA and computes a reward by matching the responses to ground truth answers directly without
    using BERTScore. Prompt metadata must have "questions" and "answers" keys. See
    https://github.com/kvablack/LLaVA-server for server-side code.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 4
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadata_batched = np.array_split(metadata, np.ceil(len(metadata) / batch_size))

        all_scores = []
        all_info = {
            "answers": [],
        }
        for image_batch, metadata_batch in zip(images_batched, metadata_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [m["questions"] for m in metadata_batch],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            correct = np.array(
                [
                    [ans in resp for ans, resp in zip(m["answers"], responses)]
                    for m, responses in zip(metadata_batch, response_data["outputs"])
                ]
            )
            scores = correct.mean(axis=-1)

            all_scores += scores.tolist()
            all_info["answers"] += response_data["outputs"]

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn


def llava_bertscore():
    """Submits images to LLaVA and computes a reward by comparing the responses to the prompts using BERTScore. See
    https://github.com/kvablack/LLaVA-server for server-side code.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 16
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        all_info = {
            "precision": [],
            "f1": [],
            "outputs": [],
        }
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [["Answer concisely: what is going on in this image?"]]
                * len(image_batch),
                "answers": [
                    [f"The image contains {prompt}"] for prompt in prompt_batch
                ],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            # use the recall score as the reward
            scores = np.array(response_data["recall"]).squeeze()
            all_scores += scores.tolist()

            # save the precision and f1 scores for analysis
            all_info["precision"] += (
                np.array(response_data["precision"]).squeeze().tolist()
            )
            all_info["f1"] += np.array(response_data["f1"]).squeeze().tolist()
            all_info["outputs"] += np.array(response_data["outputs"]).squeeze().tolist()

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn


def geometric_vanishing_point_fast(K=None, num_samples=50, min_length=20.0, threshold_c=0.03, min_support=3):
    """
    Fast geometric reward based on vanishing point detection using intersection sampling.
    
    Returns BINARY reward: 1.0 if finds VP with sufficient line support, 0.0 otherwise.
    This is much faster than the full SDP solver (~5-10x speedup).
    
    Args:
        K: Camera intrinsic matrix (3x3 numpy array). If None, auto-estimated from image size.
           Format: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        num_samples: Number of random line pairs to sample for VP candidates (default: 50).
                     Higher values are more thorough but slower. 30 is a good balance.
        min_length: Minimum line segment length in pixels for LSD detection (default: 20.0).
        threshold_c: Consistency threshold for line-VP alignment (default: 0.03).
                     Lower values require stricter alignment.
        min_support: Minimum number of lines needed to count as success (default: 3).
    
    Returns:
        Reward function compatible with DDPO framework.
        Rewards are binary: 1.0 (success) or 0.0 (failure).
    """
    from ddpo_pytorch.utils.geometric_rewards import ImageGeometricReward
    
    reward_calculator = ImageGeometricReward(K=K, min_length=min_length)
    
    def _fn(images, prompts, metadata):
        # Convert torch tensor to numpy if needed
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        
        rewards = []
        failed_count = 0
        
        for image in images:
            try:
                reward = reward_calculator.get_intersection_sampling_reward(
                    image, num_samples=num_samples, threshold_c=threshold_c, min_support=min_support
                )
                rewards.append(reward)
            except Exception as e:
                # If line detection fails (e.g., no lines detected), assign zero reward
                rewards.append(0.0)
                failed_count += 1
        
        info = {"failed_detections": failed_count} if failed_count > 0 else {}
        return np.array(rewards), info
    
    return _fn


def geometric_vanishing_point_full(K=None, min_length=20.0, threshold_c=0.03, solver_timeout=1.0, max_iters=500):
    """
    Full geometric reward using SDP-based global vanishing point optimization.
    
    Returns BINARY reward: 1.0 if SDP solver successfully converges, 0.0 otherwise.
    This encourages the model to generate images with solvable geometric structure.
    
    This uses the complete GlobustVP solver with semidefinite programming to find
    globally optimal vanishing point clusters. More accurate but slower than the fast method.
    
    Requires: cvxpy and a solver (SCS is free and included in cvxpy).
    Optional: Install GlobustVP package for full solver support.
    
    Args:
        K: Camera intrinsic matrix (3x3 numpy array). If None, auto-estimated from image size.
        min_length: Minimum line segment length in pixels for LSD detection (default: 20.0).
        threshold_c: Consistency threshold for line-VP alignment (default: 0.03).
        solver_timeout: Maximum time in seconds for SDP solver (default: 1.0).
        max_iters: Maximum solver iterations (default: 500).
    
    Returns:
        Reward function compatible with DDPO framework.
        Rewards are binary: 1.0 (solver succeeds) or 0.0 (solver fails).
    """
    from ddpo_pytorch.utils.geometric_rewards import ImageGeometricReward
    
    reward_calculator = ImageGeometricReward(K=K, min_length=min_length)
    
    def _fn(images, prompts, metadata):
        # Convert torch tensor to numpy if needed
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        
        rewards = []
        failed_count = 0
        
        for image in images:
            try:
                reward = reward_calculator.get_max_consensus_reward(
                    image, solver_timeout=solver_timeout, threshold_c=threshold_c, max_iters=max_iters
                )
                rewards.append(reward)
            except ImportError:
                raise ImportError(
                    "Full SDP solver requires GlobustVP package. "
                    "Install with: pip install -e /path/to/GlobustVP\n"
                    "Or use 'geometric_vanishing_point_fast' instead."
                )
            except Exception as e:
                # If line detection or solver fails, assign zero reward
                rewards.append(0.0)
                failed_count += 1
        
        info = {"failed_detections": failed_count} if failed_count > 0 else {}
        return np.array(rewards), info
    
    return _fn

def geometric_algebraic(num_samples=5, min_length=20.0, threshold_c=0.03):
    """
    Geometric reward based on algebraic intersection quality.
    Uses GPU-accelerated tensor operations for distance calculation.
    """
    from ddpo_pytorch.utils.geometric_rewards import ImageGeometricReward
    
    reward_calculator = ImageGeometricReward(min_length=min_length)
    
    def _fn(images, prompts, metadata):
        # Convert torch tensor to numpy if needed
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        
        rewards = []
        
        for image in images:
            try:
                reward = reward_calculator.get_algebraic_intersection_reward(
                    image, num_samples=num_samples, threshold_c=threshold_c
                )
                rewards.append(reward)
            except Exception:
                # If line detection fails, assign zero reward
                rewards.append(0.0)
        
        return np.array(rewards), {}
    
    return _fn