import cv2
import numpy as np
import os
import argparse


"""
TODO Part 1: Gamma correction
"""
def gamma_correction(img, gamma):
    """
    1. Normalize the img values from [0, 255] to [0, 1].
    2. raise each pixel value to the power of gamma
    3. Rescale the img values from [0, 1] back to [0, 255].
    """
    normalized_img = img / 255.0
    gamma_correction_img = np.power(normalized_img, gamma)
    return_img = gamma_correction_img * 255
    return_img = (gamma_correction_img * 255).astype(np.uint8)
    return return_img


"""
TODO Part 2: Histogram equalization
"""
def histogram_equalization(img):
    """
    1. change img from BGR to HSV
    2. implement histogram equalization on the V channel
    3. change the img back to BGR
    """
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image) 

    # get histogram
    hist, bins = np.histogram(v, 256, [0, 256])

    # calculate project of each pixel intensity
    sum_hist = np.cumsum(hist)
    project = np.zeros(256)
    for i in range(256):
        project[i] = (255 / sum_hist[-1]) * sum_hist[i]
    project = project.astype(np.uint8)
    v = project[v]

    equalize_hsv = cv2.merge([h, s, v])
    equalize_img = cv2.cvtColor(equalize_hsv, cv2.COLOR_HSV2BGR)

    return equalize_img


def imgae_enhance(gamma = 1.2):
    
    # read all img from 'data/test_data'
    img_dir = 'data/test_data'
    folders = [f for f in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, f))]
    
    for folder in folders:
        img_files = [f for f in os.listdir(os.path.join(img_dir, folder)) if os.path.isfile(os.path.join(img_dir, folder, f))]

        for img_file in img_files:
            img_path = os.path.join(img_dir, folder, img_file)
            # print(f"img_path: {img_path}")
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Failed to load image {img_file}")
                continue
            
            result_gamma = gamma_correction(img, gamma)
            gamma_str = str(gamma).replace('.', '_')
            gamma_folder = os.path.join(f'data/result/gamma/', folder, gamma_str)
            # print(f"gamma_folder: {gamma_folder}")
            if not os.path.exists(gamma_folder):
                # print(f"Creating folder {gamma_folder}")
                os.makedirs(gamma_folder)
            cv2.imwrite(os.path.join(gamma_folder, img_file), result_gamma)
            
            result_histogram = histogram_equalization(img)
            histogram_folder = os.path.join('data/result/histogram', folder)
            if not os.path.exists(histogram_folder):
                os.makedirs(histogram_folder)
            cv2.imwrite(os.path.join(histogram_folder, img_file), result_histogram)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Enhancement Script")
    parser.add_argument("--gamma", required=True, type=float, help="Gamma correction value")
    args = parser.parse_args()
    
    # print(f"args.gamma: {args.gamma}")
    imgae_enhance(args.gamma)
