import torch
import torchvision
import torch.optim
import os
import sys
import time
import numpy as np
from PIL import Image
import glob
import time
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import models.laplacian as model

"""
Test the model on a list of image paths. This function inputs a list
of image paths, loads the model, and outputs the enhanced images.

If the model is not parallel, use the following code to load the model:
	DCE_net = model.enhance_net_nopool().cuda()

If the model is parallel, use the following code to load the model:
	DCE_net = torch.nn.DataParallel(model.enhance_net_nopool(), device_ids=[0, 1]).cuda()

Things to change whiie each testing
	- path of the model weight
	- whether to use fuse
	- output directory 
"""

level = 4

def test(image_pyramid, output_path):

	# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
	# data_lowlight = Image.open(image_path)
	# data_lowlight = (np.asarray(data_lowlight) / 255.0)
	# data_lowlight = torch.from_numpy(data_lowlight).float()
	# data_lowlight = data_lowlight.permute(2,0,1)
	# data_lowlight = data_lowlight.cuda().unsqueeze(0)

	# Load model
	DCE_nets = []
	optimizers = []
	for i in range(level):
		# DCE_net = model.enhance_net_nopool().cuda()  # Not parallel
		DCE_net = torch.nn.DataParallel(model.enhance_net_nopool(), device_ids=[0, 1]).cuda()  # Parallel
		DCE_net.load_state_dict(torch.load(f'../snapshots/laplacian/best/{i}_Epoch199.pth'))
		DCE_nets.append(DCE_net)
	
	start = time.time()

	prev_output = None
	for i in range(level):
		DCE_net = DCE_nets[i]
		data = image_pyramid[i]

		if prev_output is not None:
			prev_output = F.interpolate(prev_output, size=(data.shape[2], data.shape[3]), mode='bilinear', align_corners=False)
			data = data + prev_output * 0.7
			data = torch.clamp(data, 0, 1)

		# Not fuse
		_, enhanced_image, _ = DCE_net(data)
		torchvision.utils.save_image(enhanced_image, f'../data/result_correct/best/{i+1}/{filename}')
		prev_output = enhanced_image

		# Fuse
		# _, enhanced_image, _ = DCE_net(1 - data_lowlight)
		# _, enhanced_image, _ = DCE_net(1 - enhanced_image)

		# Add Gaussian blur
		# enhanced_image_np = (1 - enhanced_image).squeeze().permute(1, 2, 0).cpu().numpy()
		# enhanced_image_np = cv2.GaussianBlur(enhanced_image_np, (5, 5), 0)
		# enhanced_image = torch.from_numpy(enhanced_image_np).permute(2, 0, 1).unsqueeze(0).cuda()

	print(f'Output: {output_path}')
	torchvision.utils.save_image(enhanced_image, output_path)

if __name__ == '__main__':
	with torch.no_grad():
		test_dir_path = f'../data/pyramid/test/{level}_level/'
		dir_list = os.listdir(test_dir_path)
		print(dir_list)

		for dir_name in dir_list:
			filepath_list = glob.glob(test_dir_path + dir_name + '/0/*')
			for filepath in filepath_list:  # For file in DICM, LIME, ...
				filename = filepath.split('/')[-1]

				# Load the image pyramid
				image_pyramid = []
				for i in range(level):
					pyramid_filepath = filepath.replace('0', str(i), 1)
					img = Image.open(pyramid_filepath)
					img = (np.asarray(img) / 255.0)
					img = torch.from_numpy(img).float()
					img = img.permute(2,0,1)
					img = img.cuda().unsqueeze(0)
					image_pyramid.append(img)
		
				# image_pyramid = image_pyramid[::-1]
				print(f'Input: {filepath}')
				output_path = f'../data/result_correct/best/{dir_name}/{filename}'
				test(image_pyramid, output_path)	
			break

		

