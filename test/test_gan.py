import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time

# 加入專案根目錄到 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import utils.dataloader as dataloader
import models.gan as model_gan
import models.backword as model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time

def lowlight(image_path, generator_path):
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	
	# Load and preprocess image
	data_lowlight = Image.open(image_path)
	data_lowlight = (np.asarray(data_lowlight)/255.0)
	data_lowlight = torch.from_numpy(data_lowlight).float()
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)

	# Load generator model (using the correct path)
	generator = model.enhance_net_nopool().cuda()
	generator.load_state_dict(torch.load(generator_path))
	generator.eval()

	# Process image
	start = time.time()
	with torch.no_grad():
		_, enhanced_image, _ = generator(data_lowlight)
	
	process_time = time.time() - start
	print(f'Processing time: {process_time:.4f} seconds')

	# Save result
	result_path = image_path.replace('test_data', 'result')
	os.makedirs(os.path.dirname(result_path), exist_ok=True)
	torchvision.utils.save_image(enhanced_image, result_path)
	
	return result_path

def test_batch(test_dir, generator_path):
	# Verify that we're using the generator weights, not discriminator weights
	if 'discriminator' in generator_path:
		generator_path = generator_path.replace('discriminator', 'generator')
		print(f"Warning: Switched to generator weights at: {generator_path}")
	
	print(f"Testing GAN-based model with weights from: {generator_path}")
	print(f"Processing images from: {test_dir}")
	
	# Verify the generator weights exist
	if not os.path.exists(generator_path):
		raise FileNotFoundError(f"Generator weights not found at: {generator_path}")
	
	# Create results directory if it doesn't exist
	results_dir = test_dir.replace('test_data', 'result_gan')
	os.makedirs(results_dir, exist_ok=True)
	
	# Process all images in directory
	with torch.no_grad():
		image_files = glob.glob(os.path.join(test_dir, '*.*'))
		total_images = len(image_files)
		
		print(f"Found {total_images} images to process")
		
		for idx, image_path in enumerate(image_files, 1):
			try:
				print(f"Processing image {idx}/{total_images}: {image_path}")
				result_path = lowlight(image_path, generator_path)
				print(f"Saved enhanced image to: {result_path}")
			except Exception as e:
				print(f"Error processing {image_path}: {str(e)}")

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--test_dir', type=str, default='data/test_data/Over',
						help='directory containing test images')
	parser.add_argument('--generator_path', type=str, 
						default='snapshots/weights_gan/gan_generator.pth',
						help='path to trained generator weights')
	args = parser.parse_args()

	# Test the model
	with torch.no_grad():
		test_batch(args.test_dir, args.generator_path)

