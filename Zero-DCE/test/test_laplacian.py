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

def test(image_path):
	best_level = 4

	# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
	data_lowlight = Image.open(image_path)
	data_lowlight = (np.asarray(data_lowlight) / 255.0)
	data_lowlight = torch.from_numpy(data_lowlight).float()
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)

	# Load model
	DCE_nets = []
	optimizers = []
	for i in range(best_level):
		# DCE_net = model.enhance_net_nopool().cuda()  # Not parallel
		DCE_net = torch.nn.DataParallel(model.enhance_net_nopool(), device_ids=[0, 1]).cuda()  # Parallel
		DCE_net.load_state_dict(torch.load(f'../snapshots/laplacian/{i}_Epoch199.pth'))
		DCE_nets.append(DCE_net)
	
	start = time.time()

	# Not fuse
	_, enhanced_image, _ = DCE_net(data_lowlight)

	# Fuse
	# _, enhanced_image, _ = DCE_net(1 - data_lowlight)
	# _, enhanced_image, _ = DCE_net(1 - enhanced_image)

	# Add Gaussian blur
	# enhanced_image_np = (1 - enhanced_image).squeeze().permute(1, 2, 0).cpu().numpy()
	# enhanced_image_np = cv2.GaussianBlur(enhanced_image_np, (5, 5), 0)
	# enhanced_image = torch.from_numpy(enhanced_image_np).permute(2, 0, 1).unsqueeze(0).cuda()

	image_path = image_path.replace('test_data','result/laplacian_test')
	result_path = image_path
	if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
		os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))

	print(f'Output: {result_path}')
	torchvision.utils.save_image(enhanced_image, result_path)

if __name__ == '__main__':
	with torch.no_grad():
		file_path = '../data/test_data/'
		file_list = os.listdir(file_path)

		for file_name in file_list:
			test_list = glob.glob(file_path + file_name + "/*") 
			for image_path in test_list:
				print(f'Input: {image_path}')
				test(image_path)

		

