#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
from sys import argv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import os
from keras.models import load_model
from keras.utils import plot_model, np_utils
from sklearn.metrics import confusion_matrix
import keras.backend as K
import load_data
from lime import lime_image
from skimage.segmentation import slic
from keras.utils import np_utils




# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def predict(inputs):
    pass
#     print('hihihi')
#     print('inputs shape:', inputs.shape)
    # Input: image tensor
    # Returns a predict function which returns the probabilities of labels ((7,) numpy array)
    # ex: return model(data).numpy()
    # TODO:
    img = []
    for line in inputs:
        img.append([])
        for i in range(48):
            img[-1].append([])
            for j in range(48):
                img[-1][-1].append(line[i][j][0])
    img = np.array(img)
    img = img.reshape(-1, 48, 48, 1)
#     print('img shape:', img.shape)
#     input()
    pred = model.predict(img)
#     print(pred)
#     print('ff')
#     input()
    return pred

def segmentation(inputs):
    pass
    # Input: image numpy array
    # Returns a segmentation function which returns the segmentation labels array ((48,48) numpy array)
    # ex: return skimage.segmentation.slic()
    # TODO:
#     input = input[0]
#     input = input.reshape(48, 48)
#     print('seg input shape:', inputs.shape)
    img = slic(inputs, n_segments=100, compactness=10.0)
    img = np.array(img)
#     for i in img:
#         print(i)
#     print('here, ', img.shape)
    return img

def lime_load(text  , train = 1):
    file = pd.read_csv(text).as_matrix()
    data = []
    for i in range(file.shape[0]) :
        data.extend(file[i,1].split())

    data = np.array(data).reshape(file.shape[0],48, 48).astype('float')
#     print(data.shape)
#     input()
    img = []
    for line in data:
        img.append([])
        for i in range(48):
            img[-1].append([])
            for j in range(48):
                a = [line[i][j], line[i][j], line[i][j]]
                img[-1][-1].append(a)
    img = np.array(img)
#     print(img.shape)
#     input()
    if train == 1 :
        target = file[:,0]
        print ('train loaded done...')
        return  img , target

    print ('test loaded done...')
    return  img

# In[2]:


CATEGORY = 7
SHAPE = 48

STRUCTURE = 1
SALIENCY = 1
GDA = 1
LIME = 1
FILTER_OUTPUT = 1


# argv: [1]train.csv [2]figure save path
save_folder = sys.argv[2]


print('load data...')
X_train , Y_train = load_data.load(sys.argv[1] , 1)
X_train = X_train.reshape(-1, 48, 48, 1)
#### rescale
X_train = X_train / 255.

Y_train = np_utils.to_categorical(Y_train, 7)


x_train = []
y_train = []
x_val = []
y_val = []

#### split validation set, hyperParameters: d_set, remander_value
for i, n in enumerate(X_train):
    if i%10 == 0:
        x_val.append(n)
        y_val.append(Y_train[i])
    else:
        x_train.append(n)
        y_train.append(Y_train[i])
X_val = np.array(x_val)
Y_val = np.array(y_val)
X_train = np.array(x_train)
Y_train = np.array(y_train)
print(X_val.shape)

#### plot on validation data
X = X_val
Y = Y_val


print("load model...")
model_name = 'model_0.h5'
model = load_model(model_name)

label = ["angry", "disgust", "fear", "happy", "sad", "suprise", "neutral"]


# In[3]:


# emotion_classifier = load_model(model_name)
# preds = emotion_classifier.predict(X_val)
# preds = np.argmax(preds , axis = 1) 
# yy = np.argmax(Y , axis = 1) 
# for i in range(len(X_val)):
#     if preds[i] == yy[i]:
#         print(str(i), Y[i])


# In[4]:


if STRUCTURE:
    model.summary()
    print("plot structure...")
    #plot_model(model, show_layer_names=False, show_shapes=True, to_file=model_name[:-3] + "_struct.png")

if SALIENCY:
    plt.clf()
    print("print saliency map...")
#     plt.figure(figsize=(16, 6))
    emotion_classifier = load_model(model_name)
    input_img = emotion_classifier.input
    img_ids = [(1, 95), (2, 583), (3, 50), (4, 577), (5, 208), (6, 587), (7, 274)]
    for i, idx in img_ids:
        print("plot figure %d." % idx)
        img = X[idx].reshape(1, 48, 48, 1)
        val_proba = emotion_classifier.predict(img)
        pred = val_proba.argmax(axis=-1)
        pred = np.int32(pred)
        a = pred[0]
        #print(emotion_classifier.output[:, np.int32(a)])
        target = K.mean(emotion_classifier.output[:, np.int32(a)])
        grads = K.gradients(target, input_img)[0]
        fn = K.function([input_img, K.learning_phase()], [grads])
        
        heatmap = fn([img, 0])[0]
        heatmap = heatmap.reshape(48, 48)
        heatmap /= heatmap.std()
        
        plt.imsave( os.path.join(save_folder, 'fig1_'+str(i-1)+'.jpg'), abs(heatmap), cmap='jet' )
#         plt.imshow(abs(heatmap), cmap='jet')
#         plt.savefig(os.path.join(save_folder, 'fig1_'+str(i-1)+'.jpg'))
#         see = img.reshape(48, 48)
#         plt.subplot(3, 7, i)
#         plt.imshow(see, cmap='gray')
#         plt.title("%d. %s" % (idx, label[Y[idx].argmax()]) )
        
#         thres = heatmap.std()
#         see[np.where(abs(heatmap) <= thres)] = np.mean(see)

#         plt.subplot(3, 7, i+7)
#         plt.imshow(heatmap, cmap='jet')
#         plt.colorbar()
#         plt.tight_layout()
        
#         plt.subplot(3, 7, i+14)
#         plt.imshow(see,cmap='gray')
#         plt.colorbar()
#         plt.tight_layout()

#     plt.savefig("%s_sm.png" % model_name[:-3], dpi=100)
#     plt.show()

if FILTER_OUTPUT:
#     print(model.layers)
#     input()
    img = X[2800].reshape(1, 48, 48, 1)
    get_3rd_layer_output = K.function([model.layers[0].input], [model.layers[1].output])
    layer_output = get_3rd_layer_output([img])[0]
    plt.clf()
#     print(layer_output.shape)
    for i in range(32):
        filter_output = layer_output[0, :, :, i].reshape(44, 44)
        plt.subplot(4, 8, i+1)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.imshow(filter_output, cmap='gray')

    plt.savefig(os.path.join(save_folder, 'fig2_2.jpg'))


if GDA:
    
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    input_img = model.input
    layer_name = "conv2d_1"
    print("process on layer " + layer_name)
    filter_index = range(32)
    plt.clf()
    # for loop
    random_img = np.random.random((1, 48, 48, 1))
    for f in filter_index:
        print("process on filter " + repr(f))
        layer_output = layer_dict[layer_name].output

        loss = K.mean(layer_output[:, :, :, f])
        grads = K.gradients(loss, input_img)[0]
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        iterate = K.function([input_img], [loss, grads])

        input_img_data = np.array(random_img)

        step = 1.
        for i in range(30):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step
            print("\riteration: " + repr(i) + ", current loss: " + repr(loss_value), end="", flush=True)
            if loss_value <= 0:
                break
        print("", flush=True)
        img = input_img_data[0].reshape(48, 48)
        img = deprocess_image(img)
        plt.subplot(4, 8, f+1)
        plt.title(repr(f))
        plt.xticks([], [])
        plt.yticks([], [])
        plt.imshow(input_img_data[0].reshape(48, 48), cmap='jet')

    plt.savefig(os.path.join(save_folder, 'fig2_1.jpg'))
        

if LIME:
#     idx = 577
    img_ids = [(1, 950), (2, 5830), (3, 500), (4, 5770), (5, 2080), (6, 5870), (7, 2740)]


    x_train_rgb,  x_label  = lime_load(sys.argv[1]) # for lime
    print(x_train_rgb.shape)
    # input()

    x_train_rgb = x_train_rgb / 255.
    plt.clf()
    for i, idx in img_ids:
        
        # Initiate explainer instance
        explainer = lime_image.LimeImageExplainer()

        # input()
        #p = predict(x_train_rgb[idx])
        # Get the explaination of an image
        explaination = explainer.explain_instance(
                                    image=x_train_rgb[idx], 
                                    classifier_fn=predict,
                                    segmentation_fn=segmentation
                                )

        print('part1 done...')
        # Get processed image
        images, mask = explaination.get_image_and_mask(
                                        label=x_label[idx],
                                        positive_only=False,
                                        hide_rest=False,
                                        num_features=5,
                                        min_weight=0.0
                                    )

        # save the image
        plt.imsave(os.path.join(save_folder, 'fig3_'+str(i-1)+'.jpg'), images)
        print('image: '+str(idx)+' Lime image saved..')

    
    
    
    
    
    
    
    
    
    
    