import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time


 
def lowlight(image_path, weight_TV=200, weight_spa=1, weight_col=5, weight_exp=10):
	os.environ['CUDA_VISIBLE_DEVICES']='1'
	data_lowlight = Image.open(image_path)

 

	data_lowlight = (np.asarray(data_lowlight)/255.0)


	data_lowlight = torch.from_numpy(data_lowlight).float()
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)

	DCE_net = model.enhance_net_nopool().cuda()
	DCE_net.load_state_dict(torch.load(f'snapshots/exp{weight_exp}_col{weight_col}_Epoch190.pth'))
	start = time.time()
	_,enhanced_image,_ = DCE_net(data_lowlight)

	end_time = (time.time() - start)
	# print(end_time)
	# image_path = image_path.replace('test_data','result')
	# result_path = image_path
	# if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
	# 	os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))

	image_filename = image_path.split("/")[-1]
	result_path = f'data/result/test_weights/exp{weight_exp}_col{weight_col}_04_{image_filename}'
	print(f'{result_path} saved!')

	torchvision.utils.save_image(enhanced_image, result_path)

if __name__ == '__main__':
# test_images
	with torch.no_grad():
		# filePath = 'data/test_data/'
	
		# file_list = os.listdir(filePath)
		file_list = ['DICM/29.jpg', 'DICM/32.jpg', 'DICM/37.jpg', 'DICM/44.jpg', 'DICM/47_2.jpg', 'DICM/70.jpg']
		file_list = ['data/test_data/' + i for i in file_list]
		for weight_exp in range(10, 35, 5):
			for weight_col in range(5, 25, 5):
				for file_name in file_list:
					# test_list = glob.glob(filePath+file_name+"/*") 

					# for image in test_list:
						# image = image
						# print(image)
						lowlight(file_name, weight_col=weight_col, weight_exp=weight_exp)

		

