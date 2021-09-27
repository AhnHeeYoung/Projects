from xml.etree.ElementTree import parse
import numpy as np
#from Class_ID_Name import *
import os
import openslide

from PIL import Image

import glob

import cv2 as cv2

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader

from shapely.geometry import Polygon
from skimage import draw

import matplotlib.pyplot as plt

import multiprocessing

import pickle

import warnings

import albumentations as A

from sklearn.metrics import roc_curve, auc

import logging

from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')


# In[3]:

patch_size = 3072

all_patient_name = [i.split('/')[-1].split('.')[0] for i in sorted(glob.glob('Dict{}/Page-1/*.pickle'.format(patch_size)))] + \
                   [i.split('/')[-1].split('.')[0] for i in sorted(glob.glob('Dict{}/Page-2/*.pickle'.format(patch_size)))] + \
                    [i.split('/')[-1].split('.')[0] for i in sorted(glob.glob('Dict{}/Page-3/*.pickle'.format(patch_size)))] + \
                    [i.split('/')[-1].split('.')[0] for i in sorted(glob.glob('Dict{}/Page-4/*.pickle'.format(patch_size)))] + \
                    [i.split('/')[-1].split('.')[0] for i in sorted(glob.glob('Dict{}/Page-5/*.pickle'.format(patch_size)))] + \
                    [i.split('/')[-1].split('.')[0] for i in sorted(glob.glob('Dict{}/Page-6/*.pickle'.format(patch_size)))] + \
                    [i.split('/')[-1].split('.')[0] for i in sorted(glob.glob('Dict{}/Page-7/*.pickle'.format(patch_size)))] + \
                    [i.split('/')[-1].split('.')[0] for i in sorted(glob.glob('Dict{}/Page-8/*.pickle'.format(patch_size)))] + \
                    [i.split('/')[-1].split('.')[0] for i in sorted(glob.glob('Dict{}/Page-9/*.pickle'.format(patch_size)))] + \
                    [i.split('/')[-1].split('.')[0] for i in sorted(glob.glob('Dict{}/Page-10/*.pickle'.format(patch_size)))]



dict_path = [i for i in sorted(glob.glob('Dict{}/Page-1/*.pickle'.format(patch_size)))] + \
            [i for i in sorted(glob.glob('Dict{}/Page-2/*.pickle'.format(patch_size)))] + \
            [i for i in sorted(glob.glob('Dict{}/Page-3/*.pickle'.format(patch_size)))] + \
            [i for i in sorted(glob.glob('Dict{}/Page-4/*.pickle'.format(patch_size)))] + \
            [i for i in sorted(glob.glob('Dict{}/Page-5/*.pickle'.format(patch_size)))] + \
            [i for i in sorted(glob.glob('Dict{}/Page-6/*.pickle'.format(patch_size)))] + \
            [i for i in sorted(glob.glob('Dict{}/Page-7/*.pickle'.format(patch_size)))] + \
            [i for i in sorted(glob.glob('Dict{}/Page-8/*.pickle'.format(patch_size)))] + \
            [i for i in sorted(glob.glob('Dict{}/Page-9/*.pickle'.format(patch_size)))] + \
            [i for i in sorted(glob.glob('Dict{}/Page-10/*.pickle'.format(patch_size)))]
    
All_dict = {}



for path in dict_path:

    with open('{}'.format(path), 'rb') as ok:
        All_dict.update(pickle.load(ok))
    
    
    
    
    

class HotspotDataset(Dataset):

    def __init__(self, patient_name, All_dict, mode, patch_size):
        
        self.patient_name = patient_name
        self.All_dict = All_dict
        self.mode = mode
        self.patch_size = patch_size
        
        
        self.number_coords = 0
        for name in self.patient_name:
            self.number_coords += len(self.All_dict[name]['coords'])

            
        self.label = []
        for name in self.patient_name:
            self.label += self.All_dict[name]['label']              
            
            
        
        self.all_coords = []
        self.all_labels = []
        self.all_slides_path = []
        self.all_patient_name = []
        
        for name in self.patient_name:
            self.all_coords += self.All_dict[name]['coords']
            self.all_labels += self.All_dict[name]['label']

            for i in range(len(self.All_dict[name]['coords'])):
                self.all_slides_path += [self.All_dict[name]['slides_path']]
                self.all_patient_name += [name]      
        

        

    def __len__(self):
        return self.number_coords
    

    def __getitem__(self, idx):  
        
        slide_path = self.all_slides_path[idx]
        
        image = openslide.OpenSlide(slide_path)
        image = np.array(image.read_region((self.all_coords[idx][0], self.all_coords[idx][1]), 0, (self.patch_size, self.patch_size)))[:, :, :3]
        image = cv2.resize(image, None, fx=0.5, fy=0.5)
        
        if self.mode == 'train':
            output = A.Compose([
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        #A.Cutout(num_holes=4, max_h_size=40, max_w_size=40, p=0.5),
                        A.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.1),
                        A.RandomRotate90(p=0.5)])(image=image)
            image = output['image']    
            image = transforms.ToTensor()(image)
        
        else:
            image = transforms.ToTensor()(image)
        
        
        
        label = self.all_labels[idx]
        coords = [self.all_coords[idx][0], self.all_coords[idx][1]]
        patient_name = self.all_patient_name[idx]
        
        
        
        return {"image" : image, "label" : label, "patient_name" : patient_name, "coords" : coords, "slide_path " : slide_path}
    
    
    
    
######### Shuffle ########
all_patient_name = np.array(all_patient_name)
np.random.seed(777)
np.random.shuffle(all_patient_name)
all_patient_name = all_patient_name.tolist()
######### Shuffle ########


    
len_ = len(all_patient_name)
idx = int(len_*0.8)
    
train_patient = all_patient_name[:idx]
val_patient = all_patient_name[idx:]

    
train_dataset = HotspotDataset(train_patient, All_dict, mode='train', patch_size=patch_size)
train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=12, shuffle=True)

val_dataset = HotspotDataset(val_patient, All_dict, mode='test', patch_size=patch_size)
val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=12, shuffle=False)






# In[ ]:


run_info = 'patch_size_{}_FINAL_resize0.5'.format(patch_size)

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


# In[16]:


device = 'cuda'

model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 2, bias=True)
model.to(device)
log.info("Model Load")




#criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, eta_min=0.00001)



import torch.cuda.amp as amp  
scaler = amp.GradScaler()

# In[ ]:


num_epochs=1000

for epoch in range(num_epochs):
    
    #scheduler_cosine.step()
    log.info(' Patch Size : {} '.format(patch_size))
    log.info('\n')
    
    log.info("Epoch : {}".format(epoch+1))
    
    log.info('Train Patient : {}'.format(len(train_patient)))
    log.info('Test Patient : {}'.format(len(val_patient)))
    log.info('\n')
    
    log.info("Train -> 0 : {}개, 1 : {}개".format(np.bincount(np.array(train_dataset.label))[0], np.bincount(np.array(train_dataset.label))[1]))    
    log.info("Test ->  0 : {}개, 1 : {}개".format(np.bincount(np.array(val_dataset.label))[0], np.bincount(np.array(val_dataset.label))[1]))
    log.info('\n')
    
    train_loss_sum = []
    train_pred_label = []
    train_true_label = []
    train_pred = []
    
    model.train()
    for idx, batch in enumerate(train_dataloader):
    
        image = batch['image'].to(device)
        label = batch['label'].to(device)
        
        #with amp.autocast():
        
        output = model(image)

        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        #scaler.scale(loss).backward()
        #scaler.step(optimizer)
        #scaler.update()

        train_loss_sum += [loss.detach().cpu().tolist()]
        train_true_label += label.cpu().detach().numpy().tolist()
        train_pred_label += np.argmax(nn.Softmax(1)(output.cpu().detach()), 1)
        
        
        pred_prob = nn.Softmax(1)(output.cpu().detach()).numpy()
        train_pred += pred_prob[:, 1].tolist()
        

        true_number = np.array(train_true_label) == np.array(train_pred_label)
        train_loss = sum(train_loss_sum) / len(train_loss_sum)
        Accuracy = sum(true_number) / len(true_number)

        fpr, tpr, threshold = roc_curve(np.array(train_true_label), np.array(train_pred))
        AUC = auc(fpr, tpr)

        conf_mat = confusion_matrix(y_true = np.array(train_true_label), y_pred = np.array(train_pred_label))
        
        TN = conf_mat[0, 0]
        FP = conf_mat[0, 1]
        FN = conf_mat[1, 0]
        TP = conf_mat[1, 1]
        
        Recall = TP / (TP + FN)
        Precision = TP / (TP + FP)
        
        if (idx+1) % 100 == 0:
            log.info("Train Epoch : {}/{}, Batch : {}/{}, ACC : {:.3f}, AUC : {:.3f}, Recall : {:.3f}, Precision : {:.3f}, Loss : {:.3f}"
                     .format(epoch+1, num_epochs, idx+1, len(train_dataloader), Accuracy, AUC, Recall, Precision, train_loss))
            
            log.info("TN : {} FP : {}".format(TN, FP))
            log.info("FN : {} TP : {}".format(FN, TP))
            log.info("\n")

    log.info("Train Epoch : {}/{}, Batch : {}/{}, ACC : {:.3f}, AUC : {:.3f}, Recall : {:.3f}, Precision : {:.3f}, Loss : {:.3f}"
             .format(epoch+1, num_epochs, idx+1, len(train_dataloader), Accuracy, AUC, Recall, Precision, train_loss))
    
    log.info("0 : {}개, 1 : {}개".format(np.bincount(np.array(train_dataset.label))[0], np.bincount(np.array(train_dataset.label))[1]))
    log.info("TN : {} FP : {}".format(TN, FP))
    log.info("FN : {} TP : {}".format(FN, TP))
    
    log.info("\n")
            
            
            
            
    
    
    val_loss_sum = []
    val_pred_label = []
    val_true_label = []    
    val_pred = []
    
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    
    model.eval()    
    for idx, batch in enumerate(val_dataloader):
        
        image = batch['image'].to(device)
        label = batch['label'].to(device)
        
        #with amp.autocast():
        output = model(image)

        loss = criterion(output, label)

        val_loss_sum += [loss.detach().cpu().tolist()]
        val_true_label += label.detach().cpu().numpy().tolist()
        val_pred_label += np.argmax(nn.Softmax(1)(output.cpu().detach()), 1)
        
        
        pred_prob = nn.Softmax(1)(output.cpu().detach()).numpy()
        val_pred += pred_prob[:, 1].tolist()

        
        true_number = np.array(val_true_label) == np.array(val_pred_label)
        val_loss = sum(val_loss_sum) / len(val_loss_sum)
        Accuracy = sum(true_number) / len(true_number)


        fpr, tpr, threshold = roc_curve(np.array(val_true_label), np.array(val_pred))
        AUC = auc(fpr, tpr)

        
        if (val_true_label[idx] == 1) & (val_pred_label[idx].item() == 1):
            TP += 1
        
        if (val_true_label[idx] == 0) & (val_pred_label[idx].item() == 0):
            TN += 1
            
        if (val_true_label[idx] == 1) & (val_pred_label[idx].item() == 0):
            FN += 1
            
        if (val_true_label[idx] == 0) & (val_pred_label[idx].item() == 1):
            FP += 1
            
 
        if (idx+1) % 500 == 0:
            
            Recall = TP / (TP + FN)
            Precision = TP / (TP + FP)

            log.info("VAL Epoch : {}/{}, Batch : {}/{}, ACC : {:.3f}, AUC : {:.3f}, Recall : {:.3f}, Precision : {:.3f}, Loss : {:.3f}"
                     .format(epoch+1, num_epochs, idx+1, len(val_dataloader), Accuracy, AUC, Recall, Precision, val_loss))    
            log.info("TN : {} FP : {}".format(TN, FP))
            log.info("FN : {} TP : {}".format(FN, TP))              
            
            log.info("\n")

            
            
    Recall = TP / (TP + FN)
    Precision = TP / (TP + FP)            
            
    log.info("VAL Epoch : {}/{}, Batch : {}/{}, ACC : {:.3f}, AUC : {:.3f}, Recall : {:.3f}, Precision : {:.3f}, Loss : {:.3f}"
             .format(epoch+1, num_epochs, idx+1, len(val_dataloader), Accuracy, AUC, Recall, Precision, val_loss))
    log.info("0 : {}개, 1 : {}개".format(np.bincount(np.array(val_dataset.label))[0], np.bincount(np.array(val_dataset.label))[1]))
    log.info("TN : {} FP : {}".format(TN, FP))
    log.info("FN : {} TP : {}".format(FN, TP))   
    
    
    log.info("\n")

    torch.save(model.state_dict(), 'checkpoints/{}/Epoch {} ACC{:.3f} AUC : {:.3f} Recall : {:.3f} Precision : {:.3f} VAL Loss{:.3f}.pth'.format(run_info, epoch+1, Accuracy, AUC, Recall, Precision, val_loss))


