#!/usr/bin/env python
# coding: utf-8

# In[193]:


from cv2 import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob

from torchvision import transforms

import torch


# In[ ]:





# In[187]:


temp = np.load('temp/1_temp.npy')


# In[ ]:





# In[ ]:





# In[ ]:


from models.hovernet.net_desc import HoVerNet
net = HoVerNet(input_ch=3, nr_types=None, freeze=True, mode='original')

saved_state_dict = torch.load('weight/Epoch87_TestLoss0.8882.pth')
net.load_state_dict(saved_state_dict)
net.cuda()
net.eval()


# In[ ]:





# In[ ]:





# In[ ]:


for i in range(len(temp)):
    
    image = temp[i]
    image = transforms.ToTensor()(image)
    output = net(image.cuda().unsqueeze(0))
    
    
    output_np = output['np'][0].cpu().detach().numpy().transpose(1, 2, 0)
    
    
    
    np.save('output_{}.npy'.format(i+1), output_np)

