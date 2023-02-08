#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 23:35:48 2023

@author: anaworker
"""

from unet_model import simple_unet_model   #Use normal unet model
from keras.utils import normalize
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


image_directory = 'data/images/'
mask_directory = 'data/masks/'

SIZE = 256
image_dataset = []   
mask_dataset = []  

images = os.listdir(image_directory)

        
images = os.listdir(image_directory)
for i, image_name in enumerate(images):   
        path = image_directory+image_name
        #path = path.decode()
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (SIZE, SIZE))
        image = image/255.0 
        image_dataset.append(np.array(image))
