import argparse
import logging.handlers
import os
import sys
import datetime
import glob
import cv2
import matplotlib.pyplot as plt
import random

import numpy as np
from scipy import linalg
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

from PIL import Image

from skimage.color import rgb2gray, rgb2hed
from skimage.exposure import rescale_intensity
from skimage.util import dtype

sys.path.append('./src')
from virtualstaining.model.networks import define_D, define_G, GANLoss, get_scheduler, update_learning_rate
from virtualstaining.model.helper import StainingDataset, StainingDatasetAux


# In[2]:


def TorchRgb2hed(rgb, trans_mat):
    rgb = rgb.squeeze().permute(1, 2, 0)
    rgb = rgb + 2
    stain = -torch.log(rgb.view(-1, 3))
    stain = torch.matmul(stain, trans_mat, out=None)
    return stain.view(rgb.shape)

def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def Color_Deconvolution(Img):
    # Channel 0 -> Hema
    # Channel 1 -> Eosin
    # Channel 2 -> DAB
    Img = rgb2hed(Img)
    Img = Img[:, :, :]
    #Img = rescale_intensity(Img, out_range=(0, 1))
    # Img = 1-Img
    #Img = np.uint8(Img * 255)
    return Img


# In[ ]:





# In[ ]:





# In[107]:


run_info = 'pix2pix_lightunet_FINAL_GRAY_ALLDATA'

if not os.path.exists('checkpoints/{}'.format(run_info)):
    os.mkdir('checkpoints/{}'.format(run_info))

log = logging.getLogger('staining_log')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fileHandler = logging.FileHandler('checkpoints/{}/log.txt'.format(run_info))
streamHandler = logging.StreamHandler()
fileHandler.setFormatter(formatter)
streamHandler.setFormatter(formatter)
#
log.addHandler(fileHandler)
log.addHandler(streamHandler)



# In[403]:
ADI_path = sorted(glob.glob('../deep-stroma-score-datasets/NCT-CRC-HE-100K/ADI/*'))
BACK_path = sorted(glob.glob('../deep-stroma-score-datasets/NCT-CRC-HE-100K/BACK/*'))
DEB_path = sorted(glob.glob('../deep-stroma-score-datasets/NCT-CRC-HE-100K/DEB/*'))
LYM_path = sorted(glob.glob('../deep-stroma-score-datasets/NCT-CRC-HE-100K/LYM/*'))
MUC_path = sorted(glob.glob('../deep-stroma-score-datasets/NCT-CRC-HE-100K/MUC/*'))
MUS_path = sorted(glob.glob('../deep-stroma-score-datasets/NCT-CRC-HE-100K/MUS/*'))
NORM_path = sorted(glob.glob('../deep-stroma-score-datasets/NCT-CRC-HE-100K/NORM/*'))
STR_path = sorted(glob.glob('../deep-stroma-score-datasets/NCT-CRC-HE-100K/STR/*'))
TUM_path = sorted(glob.glob('../deep-stroma-score-datasets/NCT-CRC-HE-100K/TUM/*'))

ADI_VAL = sorted(glob.glob('../data/tissue/CRC-VAL-HE-7K/ADI/*'))
BACK_VAL = sorted(glob.glob('../data/tissue/CRC-VAL-HE-7K/BACK/*'))
DEB_VAL = sorted(glob.glob('../data/tissue/CRC-VAL-HE-7K/DEB/*'))
LYM_VAL = sorted(glob.glob('../data/tissue/CRC-VAL-HE-7K/LYM/*'))
MUC_VAL = sorted(glob.glob('../data/tissue/CRC-VAL-HE-7K/MUC/*'))
MUS_VAL = sorted(glob.glob('../data/tissue/CRC-VAL-HE-7K/MUS/*'))
NORM_VAL = sorted(glob.glob('../data/tissue/CRC-VAL-HE-7K/NORM/*'))
STR_VAL = sorted(glob.glob('../data/tissue/CRC-VAL-HE-7K/STR/*'))
TUM_VAL = sorted(glob.glob('../data/tissue/CRC-VAL-HE-7K/TUM/*'))


ADI_train_path = ADI_path
ADI_test_path = ADI_path[round(len(ADI_path)*0.8):]

BACK_train_path = BACK_path
BACK_test_path = BACK_path[round(len(BACK_path)*0.8):]

DEB_train_path = DEB_path
DEB_test_path = DEB_path[round(len(DEB_path)*0.8):]

LYM_train_path = LYM_path
LYM_test_path = LYM_path[round(len(LYM_path)*0.8):]

MUC_train_path = MUC_path
MUC_test_path = MUC_path[round(len(MUC_path)*0.8):]

MUS_train_path = MUS_path
MUS_test_path = MUS_path[round(len(MUS_path)*0.8):]

NORM_train_path = NORM_path
NORM_test_path = NORM_path[round(len(NORM_path)*0.8):]

STR_train_path = STR_path
STR_test_path = STR_path[round(len(STR_path)*0.8):]

TUM_train_path = TUM_path
TUM_test_path = TUM_path[round(len(TUM_path)*0.8):]



train_path = (ADI_train_path + BACK_train_path + DEB_train_path + LYM_train_path + MUC_train_path + MUS_train_path + NORM_train_path
             + STR_train_path + TUM_train_path)

test_path = (ADI_test_path + BACK_test_path + DEB_test_path + LYM_test_path + MUC_test_path + MUS_test_path + NORM_test_path
             + STR_test_path + TUM_test_path)

val_path = (ADI_VAL + BACK_VAL + DEB_VAL + LYM_VAL + MUC_VAL + MUS_VAL + NORM_VAL + STR_VAL + TUM_VAL)



class StainingDatasetAux(Dataset):
    def __init__(self, path):
        
        self.data_number = 5000
        self.path = path
            
    def __len__(self):

        return len(self.path)

    def __getitem__(self, idx):
        
        image = cv2.imread(self.path[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original = image.copy()
        image = transforms.ToTensor()(image)
        
        pil_image = Image.fromarray(original)
        pil_image = pil_image.convert("L")
        
        pil_image = transforms.ToTensor()(pil_image)
        pil_image = transforms.ToPILImage()(pil_image)
        pil_image = transforms.ColorJitter(contrast=0.6, brightness=0.3)(pil_image)
        pil_image = transforms.ToTensor()(pil_image)
        
        
        return {"CK_image" : image, "HE_image" : pil_image}




# In[404]:

train_dataset = StainingDatasetAux(path = train_path)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=12)

test_dataset = StainingDatasetAux(path = test_path)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=12)



#model = 'unet'
model = 'unet_light'

input_nc = 1
output_nc = 3
device= torch.device("cuda:0")
ndf = 64


# In[307]:

net_g = define_G(model, input_nc, output_nc, gpu_id=device)

net_d = define_D(input_nc + output_nc, ndf, netD='basic', gpu_id=device)


# In[309]:


lr = 0.0002
beta1 = 0.5
lr_policy='lambda'
epoch_count=1
niter=100
niter_decay=100
lr_decay_iters=50

optimizer_g = optim.Adam(net_g.parameters(),
                         lr=lr,
                         betas=(beta1, 0.999))

optimizer_d = optim.Adam(net_d.parameters(),
                             lr=lr,
                             betas=(beta1, 0.999))

criterionGAN = GANLoss().to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)
net_g_scheduler = get_scheduler(optimizer_g, lr_policy, epoch_count,
                                    niter, niter_decay, lr_decay_iters)
net_d_scheduler = get_scheduler(optimizer_d, lr_policy, epoch_count,
                                    niter, niter_decay, lr_decay_iters)

rgb_from_hed = np.array([[0.65, 0.70, 0.29],
                             [0.07, 0.99, 0.11],
                             [0.27, 0.57, 0.78]])
hed_from_rgb = linalg.inv(rgb_from_hed)
hed_from_rgb = torch.Tensor(hed_from_rgb).cuda()


# In[358]:


lamb_hed=0.9
hed_normalize=False
lamb=10
img_per_epoch =10000


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[409]:


for epoch in range(1, 50):

    loss_d_list = []
    loss_g_list = []
    loss_g_gan_list = []
    loss_g_l1_list = []
    loss_g_hed_l1_list = []
    
    net_g.train()
    net_d.train()
    for iteration, batch in enumerate(train_loader):
        real_a = batch['HE_image'].to(device).type(torch.float32)
        real_b = batch['CK_image'].to(device).type(torch.float32)
        # real_b_hed = batch['CK_bin_image'].to(device).type(torch.float32)    ######## 조정 

        # generate fake image
        fake_b = net_g(real_a)
        #        real_a.shape
        ######################
        # (1) Update D network
        ######################
        optimizer_d.zero_grad()

        # predict with fake on Discriminator and calculate loss
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False)  # if true = False

        # predict with real on Discriminator and calculate loss
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = net_d.forward(real_ab)
        loss_d_real = criterionGAN(pred_real, True)  # if true = True

        # Combined D loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5  # average Discriminator losses

        loss_d.backward()
        optimizer_d.step()

        ######################
        # (2) Update G network
        ######################

        optimizer_g.zero_grad()

        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)

        # Second, G(A) = B
        loss_g_l1 = criterionL1(fake_b, real_b)
        #real_b_hed = real_b_hed.squeeze()
        # fake_hed = TorchRgb2hed(fake_b, hed_from_rgb)

        #if hed_normalize:
        #    fake_hed -= fake_hed.min(1, keepdim=True)[0]
        #    fake_hed /= fake_hed.max(1, keepdim=True)[0]
        #    real_b_hed -= real_b_hed.min(1, keepdim=True)[0]
        #    real_b_hed /= real_b_hed.max(1, keepdim=True)[0]

        # loss_hed_l1 = criterionL1(fake_hed[:, :, :], real_b_hed[:, :, :])
        loss_g = loss_g_gan + loss_g_l1 * lamb # + loss_hed_l1 * lamb_hed
        loss_g.backward()
        optimizer_g.step()

        loss_d_list.append(loss_d.item())
        loss_g_list.append(loss_g.item())
        loss_g_gan_list.append(loss_g_gan.item())
        loss_g_l1_list.append(loss_g_l1.item())
        # loss_g_hed_l1_list.append(loss_hed_l1)

        if iteration % 100 == 0:
            log.info(
                    'Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}- GAN: {:.4f}, L1Loss: {:.4f}'.format(
                        epoch,
                        iteration,
                        len(train_loader),
                        sum(loss_d_list) / len(loss_d_list),
                        sum(loss_g_list) / len(loss_g_list),
                        sum(loss_g_gan_list) / len(loss_g_gan_list),
                        sum(loss_g_l1_list) / len(loss_g_l1_list)))

        if iteration == img_per_epoch:
            break

    update_learning_rate(net_g_scheduler, optimizer_g)
    update_learning_rate(net_d_scheduler, optimizer_d)

    # checkpoint
    if epoch % 1 == 0:
        net_g_model_out_path = "checkpoints/{}/netG_{}_epoch_{}.pth".format(
                run_info, run_info, epoch)
        net_d_model_out_path = "checkpoints/{}/netD_{}_epoch_{}.pth".format(
                run_info, run_info, epoch)
        torch.save(net_g, net_g_model_out_path)

        model_out_path = "checkpoints/{}/model_epoch_{}.pth".format(run_info,
                                                                        epoch)
        torch.save({'epoch': epoch,
                        'Generator': net_g.state_dict(),
                        'Discriminator': net_d.state_dict(),
                        'optimizer_g': optimizer_g.state_dict(),
                        'optimizer_d': optimizer_d.state_dict(),
                        'scheduler_g': net_g_scheduler.state_dict(),
                        'scheduler_d': net_d_scheduler.state_dict()
                        }, model_out_path)

        log.info('Checkpoint saved to {}'.format(run_info))
    if epoch % 200 == 0:
        today = datetime.date.today()
        net_g_model_out_path = "checkpoints/{}/{}_{}.pth".format(run_info, run_info, today.strftime("%Y%m%d"))
        torch.save(net_g.state_dict(), net_g_model_out_path)



        
############################################ TEST ############################################     
    loss_d_list = []
    loss_g_list = []
    loss_g_gan_list = []
    loss_g_l1_list = []
    loss_g_hed_l1_list = []

    net_g.eval()
    net_d.eval()
    for iteration, batch in enumerate(test_loader):
        real_a = batch['HE_image'].to(device).type(torch.float32)
        real_b = batch['CK_image'].to(device).type(torch.float32)
        # real_b_hed = batch['CK_bin_image'].to(device).type(torch.float32)    ######## 조정 

        # generate fake image
        fake_b = net_g(real_a)
        #        real_a.shape
        ######################
        # (1) Update D network
        ######################

        # predict with fake on Discriminator and calculate loss
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False)  # if true = False

        # predict with real on Discriminator and calculate loss
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = net_d.forward(real_ab)
        loss_d_real = criterionGAN(pred_real, True)  # if true = True

        # Combined D loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5  # average Discriminator losse


        ######################
        # (2) Update G network
        ######################


        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)

        # Second, G(A) = B
        loss_g_l1 = criterionL1(fake_b, real_b)
        #real_b_hed = real_b_hed.squeeze()
        # fake_hed = TorchRgb2hed(fake_b, hed_from_rgb)

        #if hed_normalize:
        #    fake_hed -= fake_hed.min(1, keepdim=True)[0]
        #    fake_hed /= fake_hed.max(1, keepdim=True)[0]
        #    real_b_hed -= real_b_hed.min(1, keepdim=True)[0]
        #    real_b_hed /= real_b_hed.max(1, keepdim=True)[0]

        # loss_hed_l1 = criterionL1(fake_hed[:, :, :], real_b_hed[:, :, :])
        loss_g = loss_g_gan + loss_g_l1 * lamb # + loss_hed_l1 * lamb_hed= 

        loss_d_list.append(loss_d.item())
        loss_g_list.append(loss_g.item())
        loss_g_gan_list.append(loss_g_gan.item())
        loss_g_l1_list.append(loss_g_l1.item())
        
        
        if iteration % 100 == 0:
            log.info(
                    'TEST Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}- GAN: {:.4f}, L1Loss: {:.4f}'.format(
                        epoch,
                        iteration,
                        len(test_loader),
                        sum(loss_d_list) / len(loss_d_list),
                        sum(loss_g_list) / len(loss_g_list),
                        sum(loss_g_gan_list) / len(loss_g_gan_list),
                        sum(loss_g_l1_list) / len(loss_g_l1_list)))


            
    if epoch % 1 == 0:
        net_g_model_out_path = "checkpoints/{}/Test netG_{}_epoch_{}.pth".format(
                run_info, run_info, epoch)
        net_d_model_out_path = "checkpoints/{}/Test netD_{}_epoch_{}.pth".format(
                run_info, run_info, epoch)
        torch.save(net_g, net_g_model_out_path)

        model_out_path = "checkpoints/{}/model_epoch_{}.pth".format(run_info,
                                                                        epoch)
        torch.save({'epoch': epoch,
                        'Generator': net_g.state_dict(),
                        'Discriminator': net_d.state_dict(),
                        'optimizer_g': optimizer_g.state_dict(),
                        'optimizer_d': optimizer_d.state_dict(),
                        'scheduler_g': net_g_scheduler.state_dict(),
                        'scheduler_d': net_d_scheduler.state_dict()
                        }, model_out_path)            
            
# In[ ]:





# In[ ]:





# In[ ]:




