"""
The dataloader used for Laplacian only (since it need to load 
multiple images for each training example)
"""

import os
import sys

import torch
import torch.utils.data as data
import torch.nn.functional as F

import numpy as np
from PIL import Image
import glob
import random
import cv2

class lowlight_loader(data.Dataset):

	def __init__(self, lowlight_images_path):

		self.train_list = []
		for i in range(len([name for name in os.listdir(lowlight_images_path)])):
			self.train_list.append(glob.glob(f'{lowlight_images_path}/{str(i)}/*.jpg'))
		self.train_list = list(map(list, zip(*self.train_list)))
		
		print("Total training examples:", len(self.train_list))

	def __getitem__(self, index):

		train_pyramid_path = self.train_list[index]
		datas = []
		for path in train_pyramid_path:
			data = Image.open(path)
			data = (np.asarray(data) / 255.0) 
			data = torch.from_numpy(data).float().permute(2,0,1)
			datas.append(data)

		return datas

	def __len__(self):
		return len(self.train_list)
