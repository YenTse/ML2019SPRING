#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import numpy as np
import torch 
import torch.nn as nn
import torchvision.transforms as transform
from torch.autograd.gradcheck import zero_gradients
from PIL import Image
import pandas as pd
import glob
import os
from matplotlib import pyplot as plt
from torchvision.utils import save_image
from torchvision.models import vgg16, vgg19, resnet50,                                resnet101, densenet121, densenet169 


# In[3]:


use_cuda = torch.cuda.is_available() # return True or False 
torch.manual_seed(8+9)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

model = resnet50(pretrained=True).to(device)
# use eval mode
model.eval()
# loss criterion
criterion = nn.CrossEntropyLoss()
trans = transform.Compose([transform.ToTensor()])


# In[4]:


IMG_PATH = sys.argv[1]
SAVE_PATH = sys.argv[2]
LABEL_PATH = './labels.csv'

def ReadFile(img_path, label_path, SAVE_PATH):
    
    #### ground truth labels
    file = pd.read_csv(label_path)
    labels = file['TrueLabel'].values
#     #### one-hot encoding
#     new_labels = []
#     for i in range(len(labels)):
#         a = np.zeros((1, 1000))
#         a[0][labels[i]] = 1
#         new_labels.append(a)
#     new_labels = np.array(new_labels)
#     print(new_labels.shape)
    new_labels = labels
    new_labels = torch.LongTensor(new_labels)
    new_labels = new_labels.unsqueeze(1)
    print(new_labels.shape)
#     input()


    epsilon = 0.089
    #### ground truth images
    img_list = glob.glob(os.path.join(img_path, '*.png'))
    for img in img_list:
        img_name = os.path.basename(img)
#         print(int(img_name[:-4]))
#         input()
        idx = int(img_name[:-4])
        image = Image.open(img)
#         print(image)
#         input()
        image = trans(image)
#         print(image.shape)
#         input()
        image = image.unsqueeze(0).to(device)
#         print(image.shape)
#         input()
        
        image.requires_grad = True
        # set gradients to zero
        zero_gradients(image)
        
        output = model(image).to(device)
#         print(output.shape)
#         input()
#         print(new_labels[idx].shape)
#         input()
#         print(output.size(0))
#         input()
#         print(new_labels[idx].size(0))
#         input()
        loss = criterion(output, new_labels[idx].to(device))
        loss.backward() 
        
        # add epsilon to image
        image = image - epsilon * image.grad.sign_()
#         image = image.squeeze(0)
#         print(image.shape)
#         input()
#         plt.imsave('results.png', image.detach().numpy())
        save_image(image, os.path.join(SAVE_PATH, img_name[:-4]+'.png'))
        print(os.path.join(SAVE_PATH, img_name[:-4]+'.png')+' saved...')
        


    
ReadFile(IMG_PATH, LABEL_PATH, SAVE_PATH)
    
    
    
    
    
    


# In[ ]:





# In[ ]:




