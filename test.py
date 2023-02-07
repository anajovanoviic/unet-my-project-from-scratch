#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 06:14:45 2023

@author: anaworker
"""

import cv2
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

image_directory = 'data/images/'
image_name = 'cju0qkwl35piu0993l0dewei2.jpg'

SIZE = 256

print(image_directory+image_name)

image = cv2.imread(image_directory+image_name)

type(image)
image.shape
plt.imshow(image)

# THIS WILL give strange coloe because opencv reads in an images in bgr,
#matplot expexts it to be rgb
image2 = cv2.imread(image_directory+image_name, cv2.IMREAD_COLOR)

image3 = Image.fromarray(image2)
image4 = image3.resize((SIZE, SIZE))
image5 = np.array(image4)

type(image2)
image2.shape
plt.imshow(image2)

plt.imshow(cv2.cvtColor(image3, cv2.COLOR_BGR2RGB))

