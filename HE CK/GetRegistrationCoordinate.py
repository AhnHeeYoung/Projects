#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

import math
import os

import cv2
import numpy as np
import openslide
from scipy import signal
from skimage import morphology
from skimage.filters import threshold_local

import torchvision
from torchvision import transforms

np.set_printoptions(threshold=np.inf)

from skimage.color import rgb2gray, rgb2hed
from skimage.exposure import rescale_intensity

from multiprocessing import Pool
import tqdm


# In[13]:


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray



def Registration_Translation(Target_Img, Moving_Img):
    Target_Img = rgb2gray(Target_Img)
    Moving_Img = rgb2gray(Moving_Img)

    adaptive_thresh = threshold_local(Target_Img, 151, offset=10)
    Target_Img_B = Target_Img < adaptive_thresh

    Target_Img_B = morphology.remove_small_objects(Target_Img_B, 200)
    # Target_Img_B = morphology.remove_small_holes(Target_Img_B, 200)
    # Target_Img_B = morphology.binary_dilation(Target_Img_B, square(3))

    adaptive_thresh = threshold_local(Moving_Img, 151, offset=10)
    Moving_Img_B = Moving_Img < adaptive_thresh
    Moving_Img_B = morphology.remove_small_objects(Moving_Img_B, 200)  # 200
    # Moving_Img_B = morphology.remove_small_holes(Moving_Img_B, 400)

    Cross_Corr = signal.fftconvolve(Target_Img_B, Moving_Img_B[::-1, ::-1])
    if np.max(Cross_Corr) != 0:
        Cross_Corr = Cross_Corr / np.max(Cross_Corr)
    else:
        Cross_Corr = Cross_Corr / 1

    ind = np.unravel_index(np.argmax(Cross_Corr, axis=None), Cross_Corr.shape)
    # Moving_Img_Translation-> + , plus the tranlation to moving image coord , Moving_Img_Translation-> - minus the tranlation to moving image coord
    Moving_Img_Translation = np.array(ind) - np.array(Moving_Img_B.shape)
    return Target_Img_B, Moving_Img_B, Moving_Img_Translation, Cross_Corr


def Local_Registration_Translation(Target_Img,
                                   Moving_Img):  # Calculate the image Translation
    Target_Img = rgb2gray(Target_Img)
    Moving_Img = rgb2gray(Moving_Img)
    #Target_Img = rgb2gray(Target_Img)
    #Moving_Img = rgb2gray(Moving_Img)
    adaptive_thresh = threshold_local(Target_Img, 51, offset=3)
    #    adaptive_thresh = threshold_local(Target_Img, 151, offset=10)
    Target_Img_B = Target_Img < adaptive_thresh
    Target_Img_B = morphology.remove_small_objects(Target_Img_B, 200)
    Target_Img_B = Target_Img_B.astype(int)
    h, w = Target_Img_B.shape
    Target_Img_B[Target_Img_B == 0] = -1

    adaptive_thresh = threshold_local(Moving_Img, 51, offset=3)
    Moving_Img_B = Moving_Img < adaptive_thresh
    Moving_Img_B = morphology.remove_small_objects(Moving_Img_B, 200)  # 200
    Moving_Img_B = Moving_Img_B.astype(int)
    h, w = Moving_Img_B.shape
    Moving_Img_B[Moving_Img_B == 0] = -1

    Cross_Corr = signal.fftconvolve(Target_Img_B, Moving_Img_B[::-1, ::-1])
    if np.max(Cross_Corr) != 0:
        Cross_Corr = Cross_Corr / np.max(Cross_Corr)
    else:
        Cross_Corr = Cross_Corr / 1

    ind = np.unravel_index(np.argmax(Cross_Corr, axis=None), Cross_Corr.shape)
    # Moving_Img_Translation-> + , plus the tranlation to moving image coord , Moving_Img_Translation-> - minus the tranlation to moving image coord
    Moving_Img_Translation = np.array(ind) - (np.array(Moving_Img_B.shape) - 1)
    return Moving_Img_Translation


def registration_map(arg):
    global CK_org_image
    global HE_org_image

    i = arg[0]
    j = arg[1]
    Global_Translation_h = arg[2]
    Global_Translation_w = arg[3]
    Local_size_w = arg[4]    # 1024
    Local_size_h = arg[5]    # 1024
    bound = arg[6]           # 2000

    HE_start_h = bound + i * Local_size_h
    HE_start_w = bound + j * Local_size_w

    CK_start_h = HE_start_h - Global_Translation_h
    CK_start_w = HE_start_w - Global_Translation_w

    HE_local_image = np.array(HE_org_image.read_region((HE_start_w, HE_start_h), 0, (2048, 2048)))[:, :, :3]
    CK_local_image = np.array(CK_org_image.read_region((CK_start_w, CK_start_h), 0, (2048, 2048)))[:, :, :3]

    local_translation = Local_Registration_Translation(HE_local_image,
                                                       CK_local_image)
    p_bg = PercentBackground(CK_local_image, 220)
    return i, j, p_bg, local_translation[0], local_translation[1], Global_Translation_h, Global_Translation_w

def Color_Deconvolution(Img, Channel):
    # Channel 0 -> Hema
    # Channel 1 -> Eosin
    # Channel 2 -> DAB
    Img = rgb2hed(Img)
    Img = Img[:, :, Channel]
    Img = rescale_intensity(Img, out_range=(0, 1))
    Img = np.uint8(Img * 255)
    return Img

def PercentBackground(Img, BG_Thres):
    if len(Img.shape) > 2:
        Img = rgb2gray(Img)
        Img = np.uint8(Img)
    White_Percent = np.mean((Img > BG_Thres))
    return White_Percent



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:

wsi_path = ['HE']+['IHC']
patient_id = [os.path.splitext(j)[0] for j in sorted(os.listdir('../../datasets/liver_fibro/liver_fibro_HE_IHC/HE'))]

dir = 'Ahn_rgb2gray(51_3)_No_Restain'
if not os.path.exists(dir):
    os.mkdir(dir)
    

for idx in range(len(patient_id)):
    
    
    ################################ Global Registration ################################
    
    Target_Slide_Path = os.path.join(wsi_path[0], patient_id[idx] + '.svs')
    Moving_Slide_Path = os.path.join(wsi_path[1], patient_id[idx] + '.svs')
    
    Target_P = openslide.open_slide(Target_Slide_Path)
    Moving_P = openslide.open_slide(Moving_Slide_Path)
    
    width_rate = Moving_P.dimensions[0] / Target_P.dimensions[0]
    height_rate = Moving_P.dimensions[1] / Target_P.dimensions[1]
    
    
    
    target_image = np.array(Target_P.read_region((0, 0), 0, Target_P.dimensions))[:, :, :3]
    moving_image = np.array(Moving_P.read_region((0, 0), 0, Moving_P.dimensions))[:, :, :3]
                            
    
    
    Target_Img = rgb2gray(target_image)
    Moving_Img = rgb2gray(moving_image)
    print("Target Img : {}".format(Target_Img.shape)) 
    print("Moving Img : {}".format(Moving_Img.shape))
    
    
    
    ################ Target Binarization ################
    adaptive_thresh = threshold_local(Target_Img, 51, offset=3)
    Target_Img_B = Target_Img < adaptive_thresh
    print("adaptive_thresh : {}".format(adaptive_thresh.shape))
    
    Target_Img_B_img = np.where(Target_Img_B == True, 1 ,0)
    Target_tensor = transforms.ToTensor()(Target_Img_B_img)
    Target_tensor = transforms.ToPILImage()(Target_tensor.float())
    Target_tensor = transforms.Resize((1000, 1000))(Target_tensor)
    Target_tensor = transforms.ToTensor()(Target_tensor)
    
    # if not os.path.exists('Binarization Image/{}'.format(patient_id[idx])):
    #    os.mkdir('Binarization Image/{}'.format(patient_id[idx]))
    #torchvision.utils.save_image(Target_tensor.float(), 'Binarization Image/{}/Target_morphology전.jpg'.format(patient_id[idx]))
    
    
    Target_Img_B = morphology.remove_small_objects(Target_Img_B, 200)
    
    Target_Img_B_img = np.where(Target_Img_B == True, 1 ,0)
    Target_tensor = transforms.ToTensor()(Target_Img_B_img)
    Target_tensor = transforms.ToPILImage()(Target_tensor.float())
    Target_tensor = transforms.Resize((1000, 1000))(Target_tensor)
    Target_tensor = transforms.ToTensor()(Target_tensor)
    #torchvision.utils.save_image(Target_tensor.float(), 'Binarization Image/{}/Target_morphology후.jpg'.format(patient_id[idx]))
    
    
    
    ################ Moving Binarization ################
    adaptive_thresh = threshold_local(Moving_Img, 51, offset=3)
    Moving_Img_B = Moving_Img < adaptive_thresh
    print("adaptive_thresh : {}".format(adaptive_thresh.shape))
    
    Moving_Img_B_img = np.where(Moving_Img_B == True, 1, 0)
    Moving_tensor = transforms.ToTensor()(Moving_Img_B_img)
    Moving_tensor = transforms.ToPILImage()(Moving_tensor.float())
    Moving_tensor = transforms.Resize((1000, 1000))(Moving_tensor)
    Moving_tensor = transforms.ToTensor()(Moving_tensor)
    #torchvision.utils.save_image(Moving_tensor.float(), 'Binarization Image/{}/Moving_morphology전.jpg'.format(patient_id[idx]))
    
    Moving_Img_B = morphology.remove_small_objects(Moving_Img_B, 200)
    
    Moving_Img_B_img = np.where(Moving_Img_B == True, 1, 0)
    Moving_tensor = transforms.ToTensor()(Moving_Img_B_img)
    Moving_tensor = transforms.ToPILImage()(Moving_tensor.float())
    Moving_tensor = transforms.Resize((1000, 1000))(Moving_tensor)
    Moving_tensor = transforms.ToTensor()(Moving_tensor)
    #torchvision.utils.save_image(Moving_tensor.float(), 'Binarization Image/{}/Moving_morphology후.jpg'.format(patient_id[idx]))
    
    
    
    ################ Cross_Corr ################
    Cross_Corr = signal.fftconvolve(Target_Img_B, Moving_Img_B[::-1, ::-1])
    if np.max(Cross_Corr) != 0:
        Cross_Corr = Cross_Corr / np.max(Cross_Corr)
    else:
        Cross_Corr = Cross_Corr / 1    
    
    #Cross_Corr_tensor = transforms.ToTensor()(Cross_Corr).float()
    #torchvision.utils.save_image(Cross_Corr_tensor, "Binarization Image/{}/Cross_Corr.jpg".format(patient_id[idx]))
    
    
    ind = np.unravel_index(np.argmax(Cross_Corr, axis=None), Cross_Corr.shape)
    Moving_Img_Translation = np.array(ind) - np.array(Moving_Img_B.shape)
    print("ind : {}".format(ind))
    print("Moving_Img_Translation : {}".format(Moving_Img_Translation))
    
    Target_Downsampled_B, Moving_Downsampled_B, G_Moving_Img_Translation = Target_Img_B, Moving_Img_B, Moving_Img_Translation
    # G_Moving_Img_Translation = G_Moving_Img_Translation * Downsample_Times
    print("G_Moving_Img_Translation : {}".format(G_Moving_Img_Translation))
    
    
    HE_org_image = Target_P
    CK_org_image = Moving_P
    
    Global_Translation = G_Moving_Img_Translation
    Slide_ID = patient_id
    Local_size_h = 2048
    Local_size_w = 2048
    print("Global Translation : {}".format(Global_Translation))
    
    [org_w, org_h] = CK_org_image.level_dimensions[0]
    bound = 1000  # delete the bound. hulistic number
    num_w = (org_w - bound * 2) // Local_size_w  # number of width
    num_h = (org_h - bound * 2) // Local_size_h  # number of height
    print("org_w : {}, org_h : {}".format(org_w, org_h))
    print("num_w : {}, num_h : {}".format(num_w, num_h))
    
    A = np.ones((num_h, num_w))
    total_len = num_w * num_h

    iter_list = [[i[0][0],
                  i[0][1],
                  Global_Translation[0],
                  Global_Translation[1],
                  Local_size_w,
                  Local_size_h,
                  bound
                  ] for i in np.ndenumerate(A)]
    
    n_workers=8
    with Pool(n_workers) as p:
        r1 = list(tqdm.tqdm(p.imap(registration_map, iter_list), total=total_len))
        
    
    np.save('{}/{}-Local-Regeistration.npy'.format(dir, patient_id[idx]), r1)
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




