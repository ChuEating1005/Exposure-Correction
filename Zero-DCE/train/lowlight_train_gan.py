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
import utils.Myloss as Myloss
import models.gan as model_gan
import models.backword as model
import numpy as np
from torchvision import transforms


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train(config):
    # Define default weights
    WEIGHT_TV = 200
    WEIGHT_SPA = 1
    WEIGHT_COL = 5
    WEIGHT_EXP = 10
    WEIGHT_ADV = 0.1

    os.environ['CUDA_VISIBLE_DEVICES']=str(config.device)

    # Initialize generator (Zero-DCE) and discriminator
    generator = model.enhance_net_nopool().cuda()
    discriminator = model_gan.Discriminator().cuda()

    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    if config.load_pretrain:
        generator.load_state_dict(torch.load(config.pretrain_dir))

    # Setup data loader
    train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)        
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, 
                                             shuffle=True, num_workers=config.num_workers, pin_memory=True)

    # Loss functions
    L_color = Myloss.L_color()
    L_spa = Myloss.L_spa()
    L_exp = Myloss.L_exp(16,0.6)
    L_TV = Myloss.L_TV()
    adversarial_criterion = nn.BCELoss()

    # Optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    generator.train()
    discriminator.train()

    for epoch in range(config.num_epochs):
        print(f'Epoch {epoch+1} of {config.num_epochs}')
        
        for iteration, img_lowlight in enumerate(train_loader):
            img_lowlight = img_lowlight.cuda()
            batch_size = img_lowlight.size(0)
            
            # Real and fake labels
            real_label = torch.ones(batch_size, 1).cuda()
            fake_label = torch.zeros(batch_size, 1).cuda()

            #########################
            # Train Discriminator
            #########################
            d_optimizer.zero_grad()
            
            # Generate enhanced image
            enhanced_image_1, enhanced_image, A = generator(img_lowlight)
            
            # Train with real images
            real_output = discriminator(img_lowlight)
            d_loss_real = adversarial_criterion(real_output, real_label)
            
            # Train with enhanced (fake) images
            fake_output = discriminator(enhanced_image.detach())
            d_loss_fake = adversarial_criterion(fake_output, fake_label)
            
            # Combined discriminator loss
            d_loss = (d_loss_real + d_loss_fake) * 0.5
            d_loss.backward()
            d_optimizer.step()

            #########################
            # Train Generator
            #########################
            g_optimizer.zero_grad()
            
            # Original Zero-DCE losses
            Loss_TV = WEIGHT_TV * L_TV(A)
            loss_spa = WEIGHT_SPA * torch.mean(L_spa(enhanced_image, img_lowlight))
            loss_col = WEIGHT_COL * torch.mean(L_color(enhanced_image))
            loss_exp = WEIGHT_EXP * torch.mean(L_exp(enhanced_image))
            
            # Adversarial loss
            fake_output = discriminator(enhanced_image)
            adversarial_loss = WEIGHT_ADV * adversarial_criterion(fake_output, real_label)
            
            # Combined generator loss
            g_loss = Loss_TV + loss_spa + loss_col + loss_exp + adversarial_loss
            g_loss.backward()
            g_optimizer.step()

            if ((iteration+1) % config.display_iter) == 0:
                print(f'[Iteration {iteration+1}] G_loss: {g_loss.item():.4f}, D_loss: {d_loss.item():.4f}')
                print(f'    Loss_TV: {Loss_TV.item():.4f}, Loss_spa: {loss_spa.item():.4f}, '
                      f'Loss_col: {loss_col.item():.4f}, Loss_exp: {loss_exp.item():.4f}, '
                      f'Loss_adv: {adversarial_loss.item():.4f}')

        if ((epoch+1) % config.snapshot_iter) == 0:
            torch.save(generator.state_dict(), 
                      f'{config.snapshots_folder}/gan_generator_Epoch{str(epoch)}.pth')
            torch.save(discriminator.state_dict(), 
                      f'{config.snapshots_folder}/gan_discriminator_Epoch{str(epoch)}.pth') 




if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/")
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=200)
	parser.add_argument('--train_batch_size', type=int, default=8)
	parser.add_argument('--val_batch_size', type=int, default=4)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=10)
	parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
	parser.add_argument('--load_pretrain', type=bool, default= False)
	parser.add_argument('--pretrain_dir', type=str, default= "snapshots/Epoch99.pth")
	parser.add_argument('--device', type=int, default=0)

	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)

	train(config)
