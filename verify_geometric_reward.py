
import numpy as np
import torch
import cv2
from ddpo_pytorch.utils.geometric_rewards import ImageGeometricReward

def create_synthetic_vp_image(size=512):
    # Create an image with lines converging to a central point
    image = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2)
    
    # Draw converging lines
    for angle in range(0, 360, 15):
        rad = np.deg2rad(angle)
        end_x = int(center[0] + size * np.cos(rad))
        end_y = int(center[1] + size * np.sin(rad))
        cv2.line(image, center, (end_x, end_y), (255, 255, 255), 2)
        
    return image

def create_random_lines_image(size=512):
    image = np.zeros((size, size, 3), dtype=np.uint8)
    for _ in range(20):
        # Random lines
        p1 = np.random.randint(0, size, 2)
        p2 = np.random.randint(0, size, 2)
        cv2.line(image, tuple(p1), tuple(p2), (255, 255, 255), 2)
    return image

def main():
    print(f"Testing Algebraic Intersection Reward on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    reward_fn = ImageGeometricReward()
    
    # Test 1: Perfect VP
    img_vp = create_synthetic_vp_image()
    reward_vp = reward_fn.get_algebraic_intersection_reward(img_vp, num_samples=100) # Use more samples for stable testing
    print(f"Perfect VP Image Reward: {reward_vp:.4f}")
    
    # Test 2: Random Lines
    img_rand = create_random_lines_image()
    reward_rand = reward_fn.get_algebraic_intersection_reward(img_rand, num_samples=100)
    print(f"Random Lines Image Reward: {reward_rand:.4f}")
    
    # Check if reward_vp > reward_rand substantially
    if reward_vp > reward_rand:
        print("SUCCESS: Converging lines yield higher reward.")
    else:
        print("WARNING: Converging lines did NOT yield higher reward.")

if __name__ == "__main__":
    main()
