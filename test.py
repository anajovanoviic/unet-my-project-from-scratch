# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 06:09:50 2023

@author: anadjj
"""
#import cv2
  
# path
path = r'C:\Users\anadjj\OneDrive - Comtrade Group\geeks14.png'

a = 5+2

print(a)
  
# Using cv2.imread() method
#img = cv2.imread(path)
  
# Displaying the image
#cv2.imshow('image', img)

import os
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from keras.utils import normalize

        
#load dataset1
   
image_directory = 'dataset1/Original/'
mask_directory = 'dataset1/Ground Truth/'


SIZE = 256
image_dataset = []
mask_dataset = []

images = os.listdir(image_directory)
images.sort()

# make a method from the below commented code for testing on one image
# def visualize image
image_name = "10.png"

# cv2.imread function in OpenCV reads an image in BGR format by default, while Image.fromarray in Pillow interprets the array as RGB by default
image = cv2.imread(image_directory+image_name, 1) 
image.shape
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))