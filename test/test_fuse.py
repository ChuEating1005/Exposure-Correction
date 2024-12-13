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
import models.fuse as model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time


 
def lowlight(image_path, weight_col=5, weight_exp=10):
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	data_lowlight = Image.open(image_path)

 

	data_lowlight = (np.asarray(data_lowlight)/255.0)


	data_lowlight = torch.from_numpy(data_lowlight).float()
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)

	DCE_net = model.enhance_net_nopool().cuda()
	DCE_net.load_state_dict(torch.load(f'snapshots/weights_fuse/exp{weight_exp}_col{weight_col}_fuse_Epoch199.pth'))
	start = time.time()
	_,enhanced_image,_ = DCE_net(data_lowlight)

	filename = image_path.split('/')[-1]
	result_path = f"data/result/test_weights3/exp{weight_exp}_col{weight_col}/"
	if not os.path.exists(result_path):
		os.makedirs(result_path)
	result_path = result_path + filename
	torchvision.utils.save_image(enhanced_image, result_path)

if __name__ == '__main__':
	filePath = 'data/test_data/Over/'
	for weight_exp in range(10, 35, 5):
		for weight_col in range(5, 25, 5):
			for file_name in filePath:
				test_list = glob.glob(filePath+file_name+"/*") 
				for image in test_list:
					print(f'Processing {image} with weight_col={weight_col}, weight_exp={weight_exp}')
					lowlight(image, weight_col=weight_col, weight_exp=weight_exp)
		

