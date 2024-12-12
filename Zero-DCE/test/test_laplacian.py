import torch
import torch.nn as nn
import torchvision
import torch.optim
import os
import sys
import time
import numpy as np
from PIL import Image
import glob
import time
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import models.laplacian as model

def lowlight(image_path):
	best_level = 4

	# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
	data_lowlight = Image.open(image_path)
	data_lowlight = (np.asarray(data_lowlight) / 255.0)
	data_lowlight = torch.from_numpy(data_lowlight).float()
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)

	DCE_nets = []
	optimizers = []
	for i in range(best_level):
		# DCE_net = model.enhance_net_nopool().cuda()
		DCE_net = torch.nn.DataParallel(model.enhance_net_nopool(), device_ids=[0, 1]).cuda()
		DCE_net.load_state_dict(torch.load(f'../snapshots/laplacian/best/{i}_Epoch199.pth'))
		DCE_nets.append(DCE_net)
	
	start = time.time()
	_, enhanced_image, _ = DCE_net(data_lowlight)
	# _, enhanced_image, _ = DCE_net(1 - data_lowlight)
	# _, enhanced_image, _ = DCE_net(1 - enhanced_image)

	# enhanced_image_np = (1 - enhanced_image).squeeze().permute(1, 2, 0).cpu().numpy()

	# enhanced_image_np = cv2.GaussianBlur(enhanced_image_np, (5, 5), 0)

	# enhanced_image = torch.from_numpy(enhanced_image_np).permute(2, 0, 1).unsqueeze(0).cuda()

	image_path = image_path.replace('test_data','result/laplacian_best')
	result_path = image_path
	if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
		os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))

	print(f'Output: {result_path}')
	torchvision.utils.save_image(enhanced_image, result_path)

if __name__ == '__main__':
	with torch.no_grad():
		filePath = '../data/test_data/'
	
		file_list = os.listdir(filePath)

		for file_name in file_list:
			test_list = glob.glob(filePath+file_name+"/*") 
			for image in test_list:
				print(image)
				lowlight(image)

		

