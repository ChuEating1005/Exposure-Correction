"""
This script constructs Gaussian and Laplacian pyramids from images in a specified directory.
It processes all images in the input directory recursively, constructs the pyramids, replace
the input base directory with the output base directory, and saves the pyramids in the output 
directory.

Supported image formats: jpg, png, jpeg, bmp, JPG, PNG, JPEG, BMP

e.g.
    input_basedir = '../data/train_data/'
    output_basedir = '../data/train_pyramid/'
    pyramidLevel = 2

    Input file structure:
        ../data/train_data/  # The input base directory
        ├── dataset1/
        │   ├── img1.jpg
        │   └── img2.png
        └── dataset2/
            ├── img3.bmp
            └── img4.bmp

    Output file structure:
        ../data/train_pyramid/  # The output base directory
        ├── dataset1/
        │   ├── 1/
        │   │   ├── img1.jpg
        │   │   └── img2.png
        │   └── 2/
        │       ├── img1.jpg
        │       └── img2.png
        └── dataset2/
            ├── 1/
            │   ├── img3.bmp
            │   └── img4.bmp
            └── 2/
                ├── img3.bmp
                └── img4.bmp
"""

import glob
import os
import cv2

# Build and return a Gaussian and Laplacian pyramid from an image.
def buildPyramid(image, levels):
    gaussianPyramid = [image]
    laplacianPyramid = []

    for i in range(levels-1):
        image = cv2.GaussianBlur(image, (5, 5), 0)
        image = cv2.pyrDown(image)
        gaussianPyramid.append(image)

    for i in range(levels-1):
        upsampled = cv2.pyrUp(gaussianPyramid[i + 1]) 
        h, w, _ = gaussianPyramid[i].shape
        upsampled = cv2.resize(upsampled, (w, h))  # Make sure the upsampled image has the same dimensions as the current level image.
        laplacian = cv2.subtract(gaussianPyramid[i], upsampled)
        laplacianPyramid.append(laplacian)

    # Reverse both pyramids so that both are in ascending order.
    laplacianPyramid.reverse()
    gaussianPyramid.reverse()

    return gaussianPyramid, laplacianPyramid

### Some variables to change  ###
input_basedir = f'../data/train_data/'
output_basedir = f'../data/train_pyramid/'
pyramidLevel = 4

# Get all images in the input_basedir recursively
image_list = []
for ext in ['jpg', 'png', 'jpeg', 'bmp', 'JPG', 'PNG', 'JPEG', 'BMP']:
    image_list += glob.glob(f'{input_basedir}/**/*.{ext}', recursive=True)

print(f'Found {len(image_list)} images in {input_basedir}')

for img_path in image_list:
    img = cv2.imread(img_path)
    gp, lp = buildPyramid(img, pyramidLevel)

    # Replace input_basedir with output_basedir and others remain the same
    output_path = img_path.replace(input_basedir, output_basedir)

    for i in range(pyramidLevel - 1):
        # The output path for i-th level of the pyramid, i+2 because the first level is from gaussian pyramid
        output_path_i = f'{os.path.dirname(output_path)}/{i+2}/{os.path.basename(img_path)}' 

        # Create directory recursively if it doesn't exist
        os.makedirs(os.path.dirname(output_path_i), exist_ok=True)

        print(f'Ouput in {output_path_i}')
        cv2.imwrite(f'{output_path_i}', lp[i])

    # The the first level is from gaussian pyramid
    output_path_i = f'{os.path.dirname(output_path)}/{1}/{os.path.basename(img_path)}'

    # Create directory recursively if it doesn't exist
    os.makedirs(os.path.dirname(output_path_i), exist_ok=True)

    print(f'Ouput in {output_path_i}')
    cv2.imwrite(output_path_i, gp[0])

print(f'Pyramid #{pyramidLevel} built!')
