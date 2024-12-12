import cv2
import numpy as np
from pathlib import Path
import os
import shutil

def check_overexposure(image_path, brightness_threshold=0.85, highlight_threshold=0.85, shadow_threshold = 0.1, shadow_ratio_limit = 0.001):
    """
    Check if an image is overexposed
    
    Parameters:
    image_path: Path to the image
    brightness_threshold: Overall brightness threshold (0-1)
    highlight_threshold: Threshold for highlight regions (0-1)
    
    Returns:
    is_overexposed: Whether the image is overexposed
    stats: A dictionary containing detailed analysis data
    """
    # Read the image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    value = hsv[:,:,2]  # Extract the brightness channel
    
    # Calculate overall brightness
    avg_brightness = np.mean(value) / 255.0
    
    # Calculate the ratio of highlight regions
    highlight_pixels = np.sum(value > 255 * highlight_threshold)
    total_pixels = value.size
    highlight_ratio = highlight_pixels / total_pixels
    
    shadow_pixels = np.sum(value < 255 * shadow_threshold)
    shadow_ratio = shadow_pixels / total_pixels
    
    # Determine if the image is overexposed
    is_overexposed = ((avg_brightness > brightness_threshold) or (highlight_ratio > 0.1))
    
    if(shadow_ratio > shadow_ratio_limit):
        is_overexposed = False
    

    if is_overexposed:
        # Create target folder (if it doesn't exist)
        os.makedirs("data/overexposed", exist_ok=True)
        
        # Copy the file to the target folder
        destination = os.path.join("data/overexposed", image_path.name)
        shutil.copy2(image_path, destination)
    # Return detailed statistics
    stats = {
        'average_brightness': avg_brightness,
        'highlight_ratio': highlight_ratio,
        'shadow_ratio' : shadow_ratio,
        'is_overexposed': is_overexposed
    }
    
    return is_overexposed, stats

def find_overexposed_images(folder_path):
    """
    Identify all overexposed images in a specified folder
    
    Parameters:
    folder_path: Path to the folder
    
    Returns:
    List of overexposed images along with their analysis results
    """
    folder = Path(folder_path)
    results = []
    
    # Supported image formats
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    # Iterate through all images
    for image_path in folder.glob("*"):
        if image_path.suffix.lower() in image_extensions:
            try:
                is_overexposed, stats = check_overexposure(image_path)
                if is_overexposed:
                    results.append({
                        'path': str(image_path),
                        'stats': stats
                    })
            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")
    
    return results

# Example usage
if __name__ == "__main__":
    folder_path = "data/train_data"
    overexposed_images = find_overexposed_images(folder_path)
    
    for image in overexposed_images:
        print(f"\nOverexposed image: {image['path']}")
        print(f"Average brightness: {image['stats']['average_brightness']:.2%}")
        print(f"Highlight region ratio: {image['stats']['highlight_ratio']:.2%}")
        print(f"Shadow region ratio: {image['stats']['shadow_ratio']:}")
    