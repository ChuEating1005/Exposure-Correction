import glob
import os
import cv2

"""
Build and return a Gaussian and Laplacian pyramid from an image.
"""
def buildPyramid(image, levels):
    # Initialize the Gaussian pyramid list.
    gaussianPyramid = [image]
    # Initialize the Laplacian pyramid list.
    laplacianPyramid = []

    # Iterate over the levels.
    for i in range(levels):
        # Apply a 5x5 Gaussian filter to the image.
        image = cv2.GaussianBlur(image, (5, 5), 0)
        # Downsample the image by a factor of 2.
        image = cv2.pyrDown(image)
        # Append the downsampled image to the Gaussian pyramid list.
        gaussianPyramid.append(image)

    # Iterate over the levels.
    for i in range(levels):
        # Upsample the image by a factor of 2.
        upsampled = cv2.pyrUp(gaussianPyramid[i + 1])
        # Get the dimensions of the current level image.
        h, w, _ = gaussianPyramid[i].shape
        # Resize the upsampled image to the dimensions of the current level image.
        upsampled = cv2.resize(upsampled, (w, h))
        # Compute the Laplacian image by subtracting the upsampled image from the current level image.
        laplacian = cv2.subtract(gaussianPyramid[i], upsampled)
        # Append the Laplacian image to the Laplacian pyramid list.
        laplacianPyramid.append(laplacian)

    return gaussianPyramid, laplacianPyramid

img_dir = '../data/train_data/'
pyramidLevel = 6

image_list = glob.glob(img_dir + '*.jpg')
for img_path in image_list:
    img = cv2.imread(img_path)
    gp, lp = buildPyramid(img, pyramidLevel)

    # Create directory if it doesn't exist
    for i in range(pyramidLevel):
        os.makedirs(f'../data/pyramid/{i}', exist_ok=True)

    # Save images
    for i in range(pyramidLevel-1):
        cv2.imwrite(f'../data/pyramid/{i}/{os.path.basename(img_path)}', lp[i])
    cv2.imwrite(f'../data/pyramid/{pyramidLevel-1}/{os.path.basename(img_path)}', gp[pyramidLevel-1])

print(f'Pyramid #{pyramidLevel} built!')
