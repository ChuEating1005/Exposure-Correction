"""
Training code using Zero-DCE + Laplacian Pyramid

Script example:
    $ python train_laplacina.py --display_iter=50 --snapshot_iter=20 --device=0

Note:
    - If you want to train the model on multiple GPUs, uncomment the line 
      with "# no parallel" and comment the line with "# parallel"
"""

import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import numpy as np
from torchvision import transforms
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import utils.laplacianDataloader as dataloader
import utils.Myloss as Myloss
import models.laplacian as model


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config, W_col=5, W_exp=10, W_tv=200, W_spa=1):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.device)

    level = config.level

    train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, 
                                               num_workers=config.num_workers, pin_memory=True)
    
    # Init loss objects
    L_color = Myloss.L_color()
    L_spa = Myloss.L_spa()
    L_exp = Myloss.L_exp(16, config.E)
    L_TV = Myloss.L_TV()

    # Create multiple models and optimizer
    DCE_nets = []
    optimizers = []
    for i in range(level):
        DCE_net = model.enhance_net_nopool().cuda()  # Not parallel
        # DCE_net = torch.nn.DataParallel(model.enhance_net_nopool(), device_ids=[0, 1])  # Parallel
        DCE_net.apply(weights_init)
        optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        DCE_nets.append(DCE_net)
        optimizers.append(optimizer)

    # Start training
    for epoch in range(config.num_epochs):
        start_time = time.time()
        for iteration, img in enumerate(train_loader):
            prev_output = None
            model_loss = []
            for i in range(level):  # Train each model
                DCE_net = DCE_nets[i]
                optimizer = optimizers[i]
                img_lowlight = img[i].cuda()

                DCE_net.train()

                # Mix input with previous output
                if prev_output is not None:
                    
                    prev_output = F.interpolate(prev_output, size=(img_lowlight.size(2), img_lowlight.size(3)), 
                                                mode='bilinear', align_corners=False)  # Upsampling the prev_output
                    img_lowlight = img_lowlight + prev_output
                    img_lowlight = torch.clamp(img_lowlight, 0, 1)

                enhanced_image_1, enhanced_image, A = DCE_net(img_lowlight)
                prev_output = enhanced_image.detach()

                # Loss Calculation
                Loss_TV = W_tv * L_TV(A)
                loss_spa = W_spa * torch.mean(L_spa(enhanced_image, img_lowlight))
                loss_col = W_col * torch.mean(L_color(enhanced_image))
                loss_exp = W_exp * torch.mean(L_exp(enhanced_image))
                loss = Loss_TV + loss_spa + loss_col + loss_exp
                model_loss.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(DCE_net.parameters(), config.grad_clip_norm)
                optimizer.step()
                DCE_net.train(mode=False)

            if ((iteration + 1) % config.display_iter) == 0:
                print(f'Loss at iteration, {iteration + 1}: {model_loss}')

        if ((epoch + 1) % config.snapshot_iter) == 0:
            for i in range(level):
                torch.save(DCE_nets[i].state_dict(), f'{config.snapshots_folder}/laplacian/{level}_new/{i+1}_E0{int(config.E/0.1)}_Epoch{str(epoch)}.pth')

        # get execution time using minutes and seconds
        execution_time = (time.time() - start_time)
        print(f"Execution time for epoch #{epoch + 1}: {int(execution_time%60)} sec")
        print("=====================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

	# Input Parameters
    parser.add_argument('--lowlight_images_path', type=str, default="../data/pyramid")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="../snapshots/")
    parser.add_argument('--load_pretrain', type=bool, default= False)
    parser.add_argument('--pretrain_dir', type=str, default= "../snapshots/Epoch99.pth")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--E', type=float, default=0.4)
    parser.add_argument('--level', type=int, default=4)
    
    config = parser.parse_args()
    
    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    train(config)