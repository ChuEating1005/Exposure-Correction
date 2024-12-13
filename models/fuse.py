import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#import pytorch_colors as colors
import numpy as np

class enhance_net_nopool(nn.Module):

	def __init__(self):
		super(enhance_net_nopool, self).__init__()

		self.relu = nn.ReLU(inplace=True)

		number_f = 32
		self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True) 
		self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
		self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
		self.e_conv7 = nn.Conv2d(number_f*2,24,3,1,1,bias=True) 

		self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
		self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

	def forward(self, x):
		# Store original input
		original_x = x.clone()
		
		# First pass - standard forward processing
		x1 = self.relu(self.e_conv1(x))
		x2 = self.relu(self.e_conv2(x1))
		x3 = self.relu(self.e_conv3(x2))
		x4 = self.relu(self.e_conv4(x3))
		
		x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
		x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))
		
		x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1)))
		r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 3, dim=1)
		
		# First enhancement path (forward)
		x_forward = original_x.clone()
		x_forward = x_forward + r1*(torch.pow(x_forward,2)-x_forward)
		x_forward = x_forward + r2*(torch.pow(x_forward,2)-x_forward)
		x_forward = x_forward + r3*(torch.pow(x_forward,2)-x_forward)
		enhance_image_1 = x_forward + r4*(torch.pow(x_forward,2)-x_forward)
		
		# Second enhancement path (backward)
		x_backward = 1 - original_x
		x_backward = x_backward + r8*(torch.pow(x_backward,2)-x_backward)
		x_backward = x_backward + r7*(torch.pow(x_backward,2)-x_backward)
		x_backward = x_backward + r6*(torch.pow(x_backward,2)-x_backward)
		x_backward = x_backward + r5*(torch.pow(x_backward,2)-x_backward)
		enhance_image_2 = 1 - x_backward
		
		# Combine both paths using adaptive weights
		alpha = torch.sigmoid(torch.mean(enhance_image_1, dim=1, keepdim=True))
		enhance_image = alpha * enhance_image_1 + (1 - alpha) * enhance_image_2
		
		# Concatenate all curve parameters
		r = torch.cat([r1,r2,r3,r4,r5,r6,r7,r8],1)
		
		return enhance_image_1, enhance_image, r



