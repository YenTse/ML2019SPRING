#!/usr/bin/env python
# coding: utf-8

# In[1]:


""" 
@ Basic Iterative LEAST-LIKELY CLASS Method 
@ Paper link: https://arxiv.org/abs/1607.02533
"""
from torchvision.models import vgg16, vgg19, resnet50,                                resnet101, densenet121, densenet169 
from torchvision.utils import save_image
from imagenet_labels import classes
from torch.autograd import Variable
from torchvision import transforms
from torchvision import models
from PIL import Image
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import math
import glob
import sys
import os


# ## hyper-parameters

# In[14]:


IMAGE_PATH = sys.argv[1]
SAVE_PATH = sys.argv[2]
LABEL_PATH = './labels.csv'
model_name = 'resnet50'
IMG_SIZE = 224
num_iter = 22
eps = 18  # positive correlation with L-infinity
alpha = 1 # gradient ascent step size, like learning rate!?

torch.manual_seed(8+9)

#### Black Box Attack, you'll get better score if you know the model behind. 
print('Model: %s\n' %(model_name))
use_cuda = torch.cuda.is_available() # return True or False 
device = 'cuda' if use_cuda else 'cpu'
print('device: ', device)


# ## get images & labels

# In[15]:


#### ground truth labels
file = pd.read_csv(LABEL_PATH)
labels = file['TrueLabel'].values
new_labels = labels
new_labels = torch.LongTensor(new_labels)
new_labels = new_labels.unsqueeze(1)


# ## main function

# In[16]:


# load model
# model = resnet50(pretrained=True).to(device)
# getattr(models, model_name) = models.model_name
model = getattr(models, model_name)(pretrained=True).to(device)

model.eval()
criterion = nn.CrossEntropyLoss()

img_list = glob.glob(os.path.join(IMAGE_PATH, '*.png'))
# print(img_list)

for img_path in img_list:

    img_name = os.path.basename(img_path)  # Desktop/mydata/images/123.jpg
    idx = int(img_name[:-4])  # int(123), to find the corresponding label

    # load image and reshape to (3, 224, 224) and RGB (not BGR)
    # preprocess as described here: http://pytorch.org/docs/master/torchvision/models.html
    orig = np.array(Image.open(img_path))
    img = orig.copy().astype(np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img /= 255.0
    img = (img - mean)/std  # image.shape=(224[0], 224[1], 3[2])
    img = img.transpose(2, 0, 1) # image.shape=(224[2], 224[0], 3[1])
    
    #### tutorial time #####
    # let orig = [[1, 2, 3], [4, 5, 6]]
    #perturbation = np.zeros_like(orig) # perturbation = [[0, 0, 0], [0, 0, 0]] 
    #perturbation = np.empty_like(orig) # perturbation = [[rd, rd, rd], [rd, rd, rd]], 'rd'=random
    #perturbation = np.full_like(x, 0.1, dtype=np.double) # perturbation = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]] 

        
    # prediction before attack
    inp = Variable(torch.from_numpy(img).to(device).float().unsqueeze(0), requires_grad=True)
    orig = torch.from_numpy(img).float().to(device).unsqueeze(0)

    out = model(inp).to(device)
    pred = np.argmax(out.data.cpu().numpy())
    print('Prediction before attack: %s' %(classes[pred].split(',')[0]))
    
    
#     inp = Variable(torch.from_numpy(img).to(device).float().unsqueeze(0), requires_grad=True)


    print('eps [%d]' %(eps))
    print('Iter [%d]' %(num_iter))
    print('alpha [%d]' %(alpha))
    print('-'*20)
    
    #### Least likely class : target
    target = np.argmin(model(inp).data.cpu().numpy())
    #### softmax : calculate class probability
    exp_target = [math.exp(i) for i in model(inp).data.cpu().numpy()[0]]
    print("The least-Likely Iter [%3d/%3d]:  Prediction: %s  Confidence: %3f"
        %(0, num_iter, classes[target].split(',')[0], \
          exp_target[target]/np.sum(exp_target)))

    for i in range(num_iter):

        ##############################################################
        out = model(inp)
        loss = criterion(out, Variable(torch.Tensor([float(target)]).to(device).long()))
        loss.backward()

        #### main algorithm 
        perturbation = (-alpha/255.0) * torch.sign(inp.grad.data)
        perturbation = torch.clamp((inp.data + perturbation) - orig, min=-eps/255.0, max=eps/255.0)
#         print('perturbation max: ', torch.max(perturbation)*255)
        inp.data = orig + perturbation
        inp.grad.data.zero_()
        ################################################################

        pred_adv = np.argmax(model(inp).data.cpu().numpy())
        exp_pred = [math.exp(i) for i in model(inp).data.cpu().numpy()[0]]
        print("Iter [%3d/%3d]:  Prediction: %s  Confidence: %3f"
                %(i, num_iter, classes[pred_adv].split(',')[0], \
                  exp_pred[pred_adv]/np.sum(exp_pred)))


        # deprocess image
        adv = inp.data.cpu().numpy()[0]
        pert = (adv-img).transpose(1,2,0)
        adv = adv.transpose(1, 2, 0)
        adv = (adv * std) + mean
        adv = adv * 255.0
        adv = np.clip(adv, 0, 255).astype(np.uint8)
        pert = pert * 255
        pert = np.clip(pert, 0, 255).astype(np.uint8)

    adv = Image.fromarray(adv)
    adv.save(os.path.join(SAVE_PATH, img_name[:-4]+'.png'))
    print(os.path.join(SAVE_PATH, img_name[:-4]+'.png')+' saved...')
    print('-'*20)

    


# In[ ]:





# In[ ]:




