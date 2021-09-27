#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import glob
import os
import pathlib
import logging

from PIL import Image

import cv2
import numpy as np
import openslide
import torch
import torch.nn as nn

import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from tifffile import memmap

import staintools

import pickle


# In[3]:


run_info = 'get_image'

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




num = 11


list_ = ['Page-{}'.format(num)]

for idx in list_:

    image_path = sorted(glob.glob('../../../../Projects/Pathology/datasets/HE_IHC/Stomach/{}/*.svs'.format(idx)))
    xml_path = sorted(glob.glob('../../../../Projects/Pathology/datasets/HE_IHC/Stomach/{}/*.xml'.format(idx)))

    patient_name = [i.split('/')[-1].split('.')[0] for i in image_path]


    for num_ in range(len(patient_name)):
        
        
        if not os.path.exists('result_npy/image/Page-{}'.format(num)):
            os.mkdir('result_npy/image/Page-{}'.format(num))
        

        patient = patient_name[num_]

        log.info('patient : {}'.format(patient))

        patient_svs_path = image_path[num_]
        patient_xml_path = xml_path[num_]


        temp = openslide.OpenSlide(patient_svs_path)

        width = temp.dimensions[0]
        height = temp.dimensions[1]

        log.info("Width : {}  Height : {}".format(width, height))

        image = temp.read_region((0, 0), 0, (width, height))
        image = np.array(image)[:, :, :3]
        image = cv2.resize(image, None, fx=1/64, fy=1/64)


        cv2.imwrite('result_npy/image/Page-{}/{}.jpg'.format(num, patient), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        log.info("Patient {}  Complete".format(patient))



