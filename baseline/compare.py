import os
import cv2
import numpy as np
from typing import List, Optional, Dict
from collections import defaultdict

def create_image_comparison(folders: List[str], compare_folder: str, prefix_length: int = 5, spacing: int = 20, texts = None):
    # Validate folders
    for folder in folders:
        if not os.path.exists(folder):
            print(f"Folder does not exist: {folder}")
            return

    # Create output folder if it doesn't exist
    if not os.path.exists(compare_folder):
        os.makedirs(compare_folder)

    # Create a dictionary to group files by prefix
    prefix_groups = defaultdict(lambda: defaultdict(str))
    
    # Collect all files with their prefixes
    for folder in folders:
        for file_name in os.listdir(folder):
            if len(file_name) >= prefix_length:
                prefix = file_name[:prefix_length]
                prefix_groups[prefix][folder] = file_name

    # Filter groups that have files from all folders
    complete_groups = {
        prefix: group_files 
        for prefix, group_files in prefix_groups.items()
        if len(group_files) == len(folders)
    }

    if not complete_groups:
        print(f"No matching files found with common {prefix_length}-character prefixes.")
        return

    def add_text_to_image(image: np.ndarray, folder_paths: List[str], texts = None) -> np.ndarray:
        """Add folder names as text labels to the combined image."""
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        font_thickness = 2
        color = (255, 255, 255)

        section_width = image.shape[1] // len(folder_paths)
        # texts = [os.path.basename(path) for path in folder_paths]

        for i, text in enumerate(texts):
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = (i * section_width) + (section_width - text_size[0]) // 2
            text_y = 60

            # Add text shadow for better visibility
            cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness + 10)
            cv2.putText(image, text, (text_x, text_y), font, font_scale, color, font_thickness)

        return image

    def load_and_resize_image(image_path: str, target_size: Optional[tuple] = None) -> Optional[np.ndarray]:
        """Load and optionally resize an image."""
        img = cv2.imread(image_path)
        if img is None:
            return None
        if target_size:
            img = cv2.resize(img, target_size)
        return img

    # Process each group of matching files
    for prefix, group_files in complete_groups.items():
        # Load first image to get dimensions
        first_folder = folders[0]
        first_file = group_files[first_folder]
        base_img = load_and_resize_image(os.path.join(first_folder, first_file))
        
        if base_img is None:
            print(f"Unable to read base image: {first_file}")
            continue

        height, width = base_img.shape[:2]
        images = [base_img]

        # Load and resize remaining images
        for folder in folders[1:]:
            file_name = group_files[folder]
            img = load_and_resize_image(os.path.join(folder, file_name), (width, height))
            if img is None:
                print(f"Unable to read image from {folder}: {file_name}")
                break
            images.append(img)

        if len(images) != len(folders):
            continue

        # Create spacing array
        spacing_array = np.ones((height, spacing, 3), dtype=np.uint8) * 255

        # Combine images with spacing
        combined = images[0]
        for img in images[1:]:
            combined = np.hstack([combined, spacing_array, img])

        # Add text labels
        combined = add_text_to_image(combined, folders, texts)
        
        save_name = f"{prefix}_comparison.jpg"
        save_path = os.path.join(compare_folder, save_name)
        cv2.imwrite(save_path, combined)
        print(f"Comparison saved: {save_path}")

    print(f"All comparisons have been saved in '{compare_folder}'.")


# Example usage
if __name__ == "__main__":
    folders_to_compare = [
        "/media/amazon/F/willi/DIP/baseline/data/result/gamma/data_dark/1_2",
        "/media/amazon/F/willi/DIP/baseline/data/result/histogram/data_dark",
        "/media/amazon/F/willi/DIP/baseline/data/result/zero_dce/data_dark",
        "/media/amazon/F/willi/DIP/baseline/data/test_data/data_dark",
        ]
    
    texts = ["gamma_1.2", "histogram", "zero_dce", "test_data"]
    output_folder = "data/compare"
    create_image_comparison(
        folders_to_compare,
        output_folder,
        prefix_length=5,  # 可以調整這個值來改變比對的前綴長度
        texts=texts,
    )