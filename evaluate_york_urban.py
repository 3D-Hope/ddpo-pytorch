
import os
import glob
import json
import numpy as np
import cv2
import torch
from tqdm import tqdm
from ddpo_pytorch.utils.geometric_rewards import ImageGeometricReward

def main():
    # Configuration
    db_path = "YorkUrbanDB"
    output_file = "york_urban_rewards.json"
    
    print(f"Scanning for images in {db_path}...")
    # Find all jpg images recursively
    # Structure is YorkUrbanDB/<ID>/<ID>.jpg
    image_paths = glob.glob(os.path.join(db_path, "*", "*.jpg"))
    image_paths.sort()
    
    if not image_paths:
        print("No images found! Check the path.")
        return

    print(f"Found {len(image_paths)} images.")
    
    # Initialize Reward Calculation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize the reward class
    # We use the same parameters as in the config
    reward_calculator = ImageGeometricReward(min_length=20.0)
    
    results = []
    rewards = []
    
    # Create visualization directory
    vis_dir = "visualizations_york"
    os.makedirs(vis_dir, exist_ok=True)
    
    print("Calculating rewards and generating visualizations...")
    viz_count = 0
    max_viz = 20  # Visualize first 20 images

    def visualize_lines(img_rgb, lines, labels, reward):
        """Draw lines colored by their cluster label."""
        vis_img = img_rgb.copy()
        
        # Colors for clusters: Red, Green, Blue, Yellow, Cyan, Magenta
        colors = [
            (0, 0, 255),    # Red (BGR)
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 255, 0),  # Cyan
            (255, 0, 255)   # Magenta
        ]
        
        # Draw lines
        if lines is not None and len(lines) > 0:
            for line, label in zip(lines, labels):
                x1, y1, x2, y2 = map(int, line)
                color = colors[label % len(colors)]
                cv2.line(vis_img, (x1, y1), (x2, y2), color, 2)
        
        # Put Reward Text
        text = f"Reward: {reward:.3f}"
        cv2.putText(vis_img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.0, (255, 255, 255), 4) # Stroke
        cv2.putText(vis_img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.0, (0, 0, 0), 2)       # Text
                    
        return vis_img

    for img_path in tqdm(image_paths):
        try:
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue
                
            # Get Reward AND internal group info
            # We need to access the internal _process_image to get labels specific to this image
            # The reward function simply returns a float, so we re-run process_image here for viz
            
            # 1. Calculate Reward
            reward = reward_calculator.get_algebraic_intersection_reward(
                img, 
                num_samples=5, 
                threshold_c=0.03
            )
            
            rewards.append(reward)
            
            # 2. Visualize (only for first N images)
            if viz_count < max_viz:
                # Re-run detection to get lines/labels for visualization
                # Note: _process_image returns (lines_2D, normalized, para_lines, uncertainty, labels)
                # lines_2D is (6, N), we need raw detection for plotting or convert back.
                # Actually _process_image expects grayscale usually but handles BGR.
                
                # We can call the public method if we expose it, or use the protected one
                # But lines_2D structure is (x1, y1, x2, y2, ...) in first 4 rows
                lines_2D, _, _, _, labels = reward_calculator._process_image(img)
                
                if lines_2D is not None:
                    # lines_2D is (6, N). 
                    # Row 0,1: x1, y1
                    # Row 3,4: x2, y2
                    # We need to stack them properly to get (N, 4)
                    p1 = lines_2D[0:2, :].T  # (N, 2)
                    p2 = lines_2D[3:5, :].T  # (N, 2)
                    raw_lines = np.hstack([p1, p2]) # (N, 4) -> x1, y1, x2, y2
                    
                    vis_img = visualize_lines(img, raw_lines, labels, reward)
                    
                    # Save
                    base_name = os.path.basename(img_path)
                    save_path = os.path.join(vis_dir, f"vis_{base_name}")
                    cv2.imwrite(save_path, vis_img)
                    viz_count += 1
            
            results.append({
                "image_path": img_path,
                "image_id": os.path.basename(os.path.dirname(img_path)),
                "reward": reward
            })
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
        
    # Calculate statistics
    if rewards:
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        min_reward = np.min(rewards)
        max_reward = np.max(rewards)
        
        print("\n" + "="*40)
        print("Results Summary")
        print("="*40)
        print(f"Total Images: {len(rewards)}")
        print(f"Mean Reward:  {mean_reward:.4f}")
        print(f"Std Dev:      {std_reward:.4f}")
        print(f"Min Reward:   {min_reward:.4f}")
        print(f"Max Reward:   {max_reward:.4f}")
        print(f"Results saved to {output_file}")
    else:
        print("No rewards calculated.")

if __name__ == "__main__":
    main()
