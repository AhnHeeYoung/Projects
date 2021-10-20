#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import logging

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


# In[9]:


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def Color_Deconvolution(Img, Channel):
    # Channel 0 -> Hema
    # Channel 1 -> Eosin
    # Channel 2 -> DAB
    Img = rgb2hed(Img)
    Img = Img[:, :, Channel]
    Img = rescale_intensity(Img, out_range=(0, 1))
    Img = np.uint8(Img * 255)
    return Img



def Registration_Translation(Target_Img, Moving_Img):
    Target_Img = rgb2gray(Target_Img)
    Moving_Img = rgb2gray(Moving_Img)

    adaptive_thresh = threshold_local(Target_Img, 51, offset=3)
    Target_Img_B = Target_Img < adaptive_thresh

    Target_Img_B = morphology.remove_small_objects(Target_Img_B, 200)
    # Target_Img_B = morphology.remove_small_holes(Target_Img_B, 200)
    # Target_Img_B = morphology.binary_dilation(Target_Img_B, square(3))

    adaptive_thresh = threshold_local(Moving_Img, 51, offset=3)
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
    Target_Img = Color_Deconvolution(Target_Img, 1)
    Moving_Img = Color_Deconvolution(Moving_Img, 2)
    #    Target_Img = rgb2gray(Target_Img)
    #    Moving_Img = rgb2gray(Moving_Img)
    adaptive_thresh = threshold_local(Target_Img, 51, offset=3)
    #  adaptive_thresh = threshold_local(Target_Img, 151, offset=10)
    Target_Img_B = Target_Img < adaptive_thresh
    Target_Img_B = morphology.remove_small_objects(Target_Img_B, 200)
    Target_Img_B = Target_Img_B.astype(int)
    h, w = Target_Img_B.shape
    Target_Img_B[Target_Img_B == 0] = -1

    #  adaptive_thresh = threshold_local(Moving_Img, 51, offset=3)
    adaptive_thresh = threshold_local(Moving_Img, 151, offset=10)
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


# In[ ]:

info = 'Binarization Image Larger3'
if not os.path.exists('Binarization Image Larger3/{}'.format(info)):
    os.mkdir('Binarization Image Larger3/{}'.format(info))
    
log = logging.getLogger('log')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fileHandler = logging.FileHandler('Binarization Image Larger3/{}_log.txt'.format(info))
streamHandler = logging.StreamHandler()
fileHandler.setFormatter(formatter)
streamHandler.setFormatter(formatter)
log.addHandler(fileHandler)
log.addHandler(streamHandler)



# In[ ]:





# In[ ]:





# In[ ]:


for idx in range(1):
    #wsi_path = ['../../datasets/liver_fibro/liver_fibro_HE_IHC/Restain/HE']+['../../datasets/liver_fibro/liver_fibro_HE_IHC/Restain/IHC']
    #patient_id = [os.path.splitext(j)[0] for j in sorted(os.listdir('../../datasets/liver_fibro/liver_fibro_HE_IHC/Restain/HE'))]
    
    patient_id = 'S06-40835'
    wsi_path = ['../../datasets/HE_IHC/HE_IHC_Slides/S06-40835']+['../../datasets/HE_IHC/HE_IHC_Slides/S06-40835']
     
    
    
    log.info("Patient_id:{}".format(patient_id[idx]))
    
    #Target_Slide_Path = os.path.join(wsi_path[0], patient_id[idx] + '.svs')
    #Moving_Slide_Path = os.path.join(wsi_path[1], patient_id[idx] + '.svs')
    
    Target_Slide_Path = os.path.join(wsi_path[0], 'HE_' + patient_id + '.svs')
    Moving_Slide_Path = os.path.join(wsi_path[1], 'LCA_' + patient_id + '.svs')    
    
    
    Target_P = openslide.open_slide(Target_Slide_Path)
    Moving_P = openslide.open_slide(Moving_Slide_Path)
    
    Downsample_Times = 32
    Level = int(math.log2(Downsample_Times)) - 1   ## Level = 4
    
    if Level >= len(Target_P.level_dimensions):
        Level = len(Target_P.level_dimensions) - 1
        
    Downsample_Times = math.ceil(Target_P.level_dimensions[0][0] // Target_P.level_dimensions[Level][0]) # Downsample_Times = 1
    log.info('Downsample_Times : {}'.format(Downsample_Times))
    
    # 전체 이미지 불러들이기.  이미지화 후, np.array() 활용하여 넘파이로 변환시켜주기.
    Target_Downsampled = np.array(Target_P.read_region((0, 0), Level, Target_P.level_dimensions[Level]))[:, :, :3]# 채널4개임.마지막은 뭐지?
    Moving_Downsampled = np.array(Moving_P.read_region((0, 0), Level, Moving_P.level_dimensions[Level]))[:, :, :3]
    log.info("Target Downsampled : {}".format(Target_Downsampled.shape))
    log.info("Moving Downsampled : {}".format(Moving_Downsampled.shape))

    
    
    
    ################################  원본 Target, Moving   Image  #################################
    Target_tensor = transforms.ToTensor()(Target_Downsampled)
    Target_tensor = transforms.ToPILImage()(Target_tensor.float())
    Target_tensor = transforms.Resize((1000, 1000))(Target_tensor)
    Target_tensor = transforms.ToTensor()(Target_tensor)    
    
    if not os.path.exists('Binarization Image Larger3/{}'.format(patient_id[idx])):
        os.mkdir('Binarization Image Larger3/{}'.format(patient_id[idx]))
    torchvision.utils.save_image(Target_tensor.float(), 'Binarization Image Larger3/{}/Target원본.jpg'.format(patient_id[idx]))        
    
    
    Moving_tensor = transforms.ToTensor()(Moving_Downsampled)
    Moving_tensor = transforms.ToPILImage()(Moving_tensor.float())
    Moving_tensor = transforms.Resize((1000, 1000))(Moving_tensor)
    Moving_tensor = transforms.ToTensor()(Moving_tensor)    
    
    if not os.path.exists('Binarization Image Larger3/{}'.format(patient_id[idx])):
        os.mkdir('Binarization Image Larger3/{}'.format(patient_id[idx]))
    torchvision.utils.save_image(Moving_tensor.float(), 'Binarization Image Larger3/{}/Moving원본.jpg'.format(patient_id[idx]))    
    
    
    
    
    
    Target_Img = rgb2gray(Target_Downsampled)
    Moving_Img = rgb2gray(Moving_Downsampled)
    log.info("Target Img : {}".format(Target_Img.shape)) 
    log.info("Moving Img : {}".format(Moving_Img.shape))
    
    
    ###### Target Img ######
    Target_tensor = transforms.ToTensor()(Target_Img)
    Target_tensor = transforms.ToPILImage()(Target_tensor.float())
    Target_tensor = transforms.Resize((1000, 1000))(Target_tensor)
    Target_tensor = transforms.ToTensor()(Target_tensor)    
    
    if not os.path.exists('Binarization Image Larger3/{}'.format(patient_id[idx])):
        os.mkdir('Binarization Image Larger3/{}'.format(patient_id[idx]))
    torchvision.utils.save_image(Target_tensor.float(), 'Binarization Image Larger3/{}/Target_Gray.jpg'.format(patient_id[idx]))    
    
    
    adaptive_thresh = threshold_local(Target_Img, 151, offset=10)
    Target_Img_B = Target_Img < adaptive_thresh
    log.info("adaptive_thresh : {}".format(adaptive_thresh.shape))
    
    Target_Img_B_img = np.where(Target_Img_B == True, 1 ,0)
    Target_tensor = transforms.ToTensor()(Target_Img_B_img)
    Target_tensor = transforms.ToPILImage()(Target_tensor.float())
    Target_tensor = transforms.Resize((1000, 1000))(Target_tensor)
    Target_tensor = transforms.ToTensor()(Target_tensor)
    
    if not os.path.exists('Binarization Image Larger3/{}'.format(patient_id[idx])):
        os.mkdir('Binarization Image Larger3/{}'.format(patient_id[idx]))
    torchvision.utils.save_image(Target_tensor.float(), 'Binarization Image Larger3/{}/Target_morphology전.jpg'.format(patient_id[idx]))
    
    
    Target_Img_B = morphology.remove_small_objects(Target_Img_B, 200)
    
    Target_Img_B_img = np.where(Target_Img_B == True, 1 ,0)
    Target_tensor = transforms.ToTensor()(Target_Img_B_img)
    Target_tensor = transforms.ToPILImage()(Target_tensor.float())
    Target_tensor = transforms.Resize((1000, 1000))(Target_tensor)
    Target_tensor = transforms.ToTensor()(Target_tensor)
    torchvision.utils.save_image(Target_tensor.float(), 'Binarization Image Larger3/{}/Target_morphology후.jpg'.format(patient_id[idx]))
    
    
    
    
    
    
    ###### Moving Img ######
    Moving_tensor = transforms.ToTensor()(Moving_Img)
    Moving_tensor = transforms.ToPILImage()(Moving_tensor.float())
    Moving_tensor = transforms.Resize((1000, 1000))(Moving_tensor)
    Moving_tensor = transforms.ToTensor()(Moving_tensor)    
    
    if not os.path.exists('Binarization Image Larger3/{}'.format(patient_id[idx])):
        os.mkdir('Binarization Image Larger3/{}'.format(patient_id[idx]))
    torchvision.utils.save_image(Moving_tensor.float(), 'Binarization Image Larger3/{}/Moving_Gray.jpg'.format(patient_id[idx]))
    
    
    adaptive_thresh = threshold_local(Moving_Img, 151, offset=10)
    Moving_Img_B = Moving_Img < adaptive_thresh
    log.info("adaptive_thresh : {}".format(adaptive_thresh.shape))

    Moving_Img_B_img = np.where(Moving_Img_B == True, 1 ,0)
    Moving_tensor = transforms.ToTensor()(Moving_Img_B_img)
    Moving_tensor = transforms.ToPILImage()(Moving_tensor.float())
    Moving_tensor = transforms.Resize((1000, 1000))(Moving_tensor)
    Moving_tensor = transforms.ToTensor()(Moving_tensor)
    torchvision.utils.save_image(Moving_tensor.float(), 'Binarization Image Larger3/{}/Moving_morphology전.jpg'.format(patient_id[idx]))
    
    Moving_Img_B = morphology.remove_small_objects(Moving_Img_B, 200)
    
    Moving_Img_B_img = np.where(Moving_Img_B == True, 1 ,0)
    Moving_tensor = transforms.ToTensor()(Moving_Img_B_img)
    Moving_tensor = transforms.ToPILImage()(Moving_tensor.float())
    Moving_tensor = transforms.Resize((1000, 1000))(Moving_tensor)
    Moving_tensor = transforms.ToTensor()(Moving_tensor)
    torchvision.utils.save_image(Moving_tensor.float(), 'Binarization Image Larger3/{}/Moving_morphology후.jpg'.format(patient_id[idx]))
    
    
    
    ## Cross_Corr ##
    Cross_Corr = signal.fftconvolve(Target_Img_B, Moving_Img_B[::-1, ::-1])
    if np.max(Cross_Corr) != 0:
        Cross_Corr = Cross_Corr / np.max(Cross_Corr)
    else:
        Cross_Corr = Cross_Corr / 1
    #Cross_Corr_tensor = transforms.ToTensor()(Cross_Corr).float()
    #torchvision.utils.save_image(Cross_Corr_tensor, "Binarization Image/{}/Cross_Corr.jpg".format(patient_id[idx]))
    
    
    
    
    
    
    ind = np.unravel_index(np.argmax(Cross_Corr, axis=None), Cross_Corr.shape)
    Moving_Img_Translation = np.array(ind) - np.array(Moving_Img_B.shape)
    log.info("ind : {}".format(ind))
    log.info("Moving_Img_Translation : {}".format(Moving_Img_Translation))
    
    Target_Downsampled_B, Moving_Downsampled_B, G_Moving_Img_Translation = Target_Img_B, Moving_Img_B, Moving_Img_Translation
    G_Moving_Img_Translation = G_Moving_Img_Translation * Downsample_Times
    print("G_Moving_Img_Translation : {}".format(G_Moving_Img_Translation))
    
    
    HE_org_image = Target_P
    CK_org_image = Moving_P
    
    Global_Translation = G_Moving_Img_Translation
    Slide_ID = patient_id
    Local_size_h = 1024
    Local_size_w = 1024
    log.info("Global Translation : {}".format(Global_Translation))
    
    [org_w, org_h] = CK_org_image.level_dimensions[0]
    bound = 5000  # delete the bound. hulistic number
    num_w = (org_w - bound * 2) // Local_size_w  # number of width
    num_h = (org_h - bound * 2) // Local_size_h  # number of height
    log.info("org_w : {}, org_h : {}".format(org_w, org_h))
    log.info("num_w : {}, num_h : {}".format(num_w, num_h))
    
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
    
    
    
    
###################################################### Local ####################################################    
    
    
    
    
    
    for iter in range(len(iter_list)):
        
        if iter>200:
            continue
        
        arg = iter_list[iter]
        
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

        HE_local_image = np.array(HE_org_image.read_region((HE_start_w, HE_start_h), 0, (1024, 1024)))[:, :, :3]
        CK_local_image = np.array(CK_org_image.read_region((CK_start_w, CK_start_h), 0, (1024, 1024)))[:, :, :3]
        #    HE_local_image = np.array(HE_org_image.read_region((HE_start_w , HE_start_h), 0, (10240, 10240)))[:, :,:3]
        #    CK_local_image = np.array(CK_org_image.read_region((CK_start_w , CK_start_h), 0, (10240, 10240)))[:, :,:3] # cut the local image 10240*10240
        
        
        
        
        # Target_Img = Color_Deconvolution(HE_local_image, 1)
        # Moving_Img = Color_Deconvolution(CK_local_image, 2)
        Target_Img = rgb2gray(HE_local_image)
        Moving_Img = rgb2gray(CK_local_image)
        
        
        
        adaptive_thresh = threshold_local(Target_Img, 151, offset=10)      
        log.info("Target block_size : 31, offset : 3")        
        # adaptive_thresh = threshold_local(Target_Img, 151, offset=10)
        Target_Img_B = Target_Img < adaptive_thresh
        Target_Img_B = morphology.remove_small_objects(Target_Img_B, 200)
        Target_Img_B = Target_Img_B.astype(int)
        h, w = Target_Img_B.shape
        #Target_Img_B[Target_Img_B == 0] = -1

        Target_Img_B_img = np.where(Target_Img_B == True, 1 ,0)
        Target_tensor = transforms.ToTensor()(Target_Img_B_img)
        Target_tensor = transforms.ToPILImage()(Target_tensor.float())
        Target_tensor = transforms.Resize((1000, 1000))(Target_tensor)
        Target_tensor = transforms.ToTensor()(Target_tensor)
        if not os.path.exists('Binarization Image Larger3/{}/Target(HE)'.format(patient_id[idx])):
            os.mkdir('Binarization Image Larger3/{}/Target(HE)'.format(patient_id[idx]))
        torchvision.utils.save_image(Target_tensor.float(), 'Binarization Image Larger3/{}/Target(HE)/{}.jpg'.format(patient_id[idx], (i, j)))
        
        
        
        
        
        
        #adaptive_thresh = threshold_local(Moving_Img, 51, offset=3)
        adaptive_thresh = threshold_local(Moving_Img, 151, offset=10)
        log.info("Target block_size : 151, offset : 10")
        Moving_Img_B = Moving_Img < adaptive_thresh
        Moving_Img_B = morphology.remove_small_objects(Moving_Img_B, 200)  # 200
        Moving_Img_B = Moving_Img_B.astype(int)
        h, w = Moving_Img_B.shape
        Moving_Img_B[Moving_Img_B == 0] = -1
                     
                     
                     
        Moving_Img_B_img = np.where(Moving_Img_B == True, 1 ,0)
        Moving_tensor = transforms.ToTensor()(Moving_Img_B_img)
        Moving_tensor = transforms.ToPILImage()(Moving_tensor.float())
        Moving_tensor = transforms.Resize((1000, 1000))(Moving_tensor)
        Moving_tensor = transforms.ToTensor()(Moving_tensor)
        if not os.path.exists('Binarization Image Larger3/{}/Moving(IHC)'.format(patient_id[idx])):
            os.mkdir('Binarization Image Larger3/{}/Moving(IHC)'.format(patient_id[idx]))                     
                     
        torchvision.utils.save_image(Moving_tensor.float(), 'Binarization Image Larger3/{}/Moving(IHC)/{}.jpg'.format(patient_id[idx], (i, j)))             
                     
                     
 

        Cross_Corr = signal.fftconvolve(Target_Img_B, Moving_Img_B[::-1, ::-1])
        if np.max(Cross_Corr) != 0:
            Cross_Corr = Cross_Corr / np.max(Cross_Corr)
        else:
            Cross_Corr = Cross_Corr / 1

        Cross_Corr_tensor = transforms.ToTensor()(Cross_Corr).float()     
        if not os.path.exists('Binarization Image Larger3/{}/Cross_Corr'.format(patient_id[idx])):
            os.mkdir('Binarization Image Larger3/{}/Cross_Corr'.format(patient_id[idx]))
                     

        ind = np.unravel_index(np.argmax(Cross_Corr, axis=None), Cross_Corr.shape)
        # Moving_Img_Translation-> + , plus the tranlation to moving image coord , Moving_Img_Translation-> - minus the tranlation to moving image coord
        Moving_Img_Translation = np.array(ind) - (np.array(Moving_Img_B.shape) - 1)                     
        
        torchvision.utils.save_image(Cross_Corr_tensor, "Binarization Image Larger3/{}/Cross_Corr/{}_ShiftVector{}.jpg".format(patient_id[idx], (i, j), Moving_Img_Translation))           
       


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




