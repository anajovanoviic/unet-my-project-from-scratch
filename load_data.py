# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 07:35:43 2023

@author: anadjj
"""

import os
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from keras.utils import normalize
from simple_unet_model import simple_unet_model

        
#load dataset1
   
image_directory = 'dataset1/Original/'
mask_directory = 'dataset1/Ground Truth/'


SIZE = 256
image_dataset = []
mask_dataset = []

images = os.listdir(image_directory)
images.sort()

#make a method from the below commented code for testing on one image
#def visualize image
# image_name = "10.png"

# # cv2.imread function in OpenCV reads an image in BGR format by default, while Image.fromarray in Pillow interprets the array as RGB by default
# image = cv2.imread(image_directory+image_name, 1) 
# image.shape
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


# Convert BGR image to RGB
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# numpy_array = np.array(image, dtype=np.uint8)
# image = Image.fromarray(numpy_array, 'RGB')
# image.show()
# image = image.resize((SIZE, SIZE))
# image_dataset.append(np.array(image))

for i, image_name in enumerate(images):
    #print(i, image_name)
    image = cv2.imread(image_directory+image_name, 1) #loading color images
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    numpy_array = np.array(image, dtype=np.uint8)
    image = Image.fromarray(numpy_array, 'RGB')
    #image.show()
    image = image.resize((SIZE, SIZE))
    image_dataset.append(np.array(image))
    
    
masks = os.listdir(mask_directory)
masks.sort()

# testing mask image
# mask_name = "1.png"
# image = cv2.imread(mask_directory+mask_name, 0)
# image = Image.fromarray(image)
# image.show()
# image = image.resize((SIZE, SIZE))
# mask_dataset.append(np.array(image))

for i, mask_name in enumerate(masks):
    image = cv2.imread(mask_directory+mask_name, 0)
    image = Image.fromarray(image)
    image = image.resize((SIZE, SIZE))
    mask_dataset.append(np.array(image))
    
# image_dataset_numpy_array = np.array(image_dataset)
# image_dataset_numpy_array_normalized = normalize(image_dataset_numpy_array, axis=1)
# #image_dataset = np.expand_dims(image_dataset_numpy_array_normalized) #i guess 3 as a parameter is not needed here 
# image_dataset = image_dataset_numpy_array_normalized   
    
image_dataset_numpy_array = np.array(image_dataset)
image_dataset_numpy_array_normalized = image_dataset_numpy_array/255.0
#image_dataset_numpy_array_normalized2 = normalize(image_dataset_numpy_array, axis=1)
image_dataset = image_dataset_numpy_array_normalized
#image_dataset = np.expand_dims(image_dataset_numpy_array_normalized, 3) 


mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.10, random_state = 0)


#Sanity check, view few mages
import random
import numpy as np
#lenox = len(X_train)
image_number = random.randint(0, 550)
#ono_sto_plotujem = np.reshape(X_train[image_number], (256, 256))
plt.figure(figsize=(12, 6))
plt.subplot(121)
z = X_train[image_number]
plt.imshow(np.reshape(X_train[image_number], (256, 256,3)), cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (256, 256,1)), cmap='gray')
plt.show()

IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH  = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]  

def get_model():
    return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = get_model()


