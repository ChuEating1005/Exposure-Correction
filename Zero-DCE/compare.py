import os
import cv2
import numpy as np
from typing import List, Optional

def create_image_comparison(folders: List[str], compare_folder: str, spacing: int = 20):
    """
    Compare images from multiple folders and create side-by-side comparisons.
    
    Args:
        folders: List of folder paths containing images to compare
        compare_folder: Output folder path for saving comparisons
        spacing: Pixels between images (default: 20)
    """
    # Validate folders
    for folder in folders:
        if not os.path.exists(folder):
            print(f"Folder does not exist: {folder}")
            return

    # Create output folder if it doesn't exist
    if not os.path.exists(compare_folder):
        os.makedirs(compare_folder)

    # Get list of files from first folder
    base_files = os.listdir(folders[0])
    
    # Find common files across all folders
    common_files = []
    for file_name in base_files:
        if all(os.path.exists(os.path.join(folder, file_name)) for folder in folders):
            common_files.append(file_name)

    if not common_files:
        print("No common files found for comparison.")
        return

    def add_text_to_image(image: np.ndarray, folder_paths: List[str]) -> np.ndarray:
        """Add folder names as text labels to the combined image."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        color = (255, 255, 255)

        section_width = image.shape[1] // len(folder_paths)
        texts = [os.path.basename(path) for path in folder_paths]

        for i, text in enumerate(texts):
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = (i * section_width) + (section_width - text_size[0]) // 2
            text_y = 30

            # Add text shadow for better visibility
            cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness + 2)
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

    # Process each common file
    for file_name in common_files:
        # Load first image to get dimensions
        base_img = load_and_resize_image(os.path.join(folders[0], file_name))
        if base_img is None:
            print(f"Unable to read base image: {file_name}")
            continue

        height, width = base_img.shape[:2]
        images = [base_img]

        # Load and resize remaining images
        for folder in folders[1:]:
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
        combined = add_text_to_image(combined, folders)

        # Save comparison
        save_path = os.path.join(compare_folder, file_name)
        cv2.imwrite(save_path, combined)
        print(f"Comparison saved: {save_path}")

    print(f"All comparisons have been saved in '{compare_folder}'.")

# Example usage
if __name__ == "__main__":
    # List of folders to compare (can be any number of folders)
    folders_to_compare = [
        "data/test_data/Over",
        "data/output1",
        "data/output2",
        "data/output3",
        # Add more folders as needed
    ]
    
    output_folder = "data/compare"
    create_image_comparison(folders_to_compare, output_folder)