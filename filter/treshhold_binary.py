#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 14:04:56 2021

@author: Anna
"""
import os 

import cv2
import matplotlib.pyplot as plt
import numpy as np

"""
Detections of connected components 
"""

os.getcwd()
os.chdir("/home/rio/Dokumente/Uni/Master/Module/Fieldecology/leaf_herbivory/images")

pathtest = "raw/IMG_4735.jpg"
img = cv2.imread(pathtest)
img = cv2.imread("raw/IMG_4769.jpg")
plt.imshow(cv2.cvtColor(img[500:2200,500:1800], cv2.COLOR_BGR2RGB))

# remove light colors from original image (background)
minBGR = np.array([0,0,0])
maxBGR = np.array([125, 255,180])
maskBGR = cv2.inRange(img, minBGR, maxBGR)
seg = cv2.bitwise_and(img, img, mask=maskBGR)

# remove non green areas
minBGR = np.array([0,65,50])
maxBGR = np.array([80, 255,255])
maskBGR = cv2.inRange(seg, minBGR, maxBGR)
crop = cv2.bitwise_and(seg, seg, mask=maskBGR)

# grayscale image 

img_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

plt.imshow(img_gray[880:3200,:], "Greys")

#img_crop = img_gray[880:3200,:]


## binary image treshold

#binary = np.zeros(maskBGR.shape, dtype = "uint8")
#for y in range(0, maskBGR.shape[1]):
#    for x in range(0,maskBGR.shape[0]):
#        if maskBGR[x,y] > 0:
#            binary[x,y] = 255

# blur (optional)

#src_gray = cv2.blur(img_gray, (3,3))
#plt.imshow(src_gray, "Greys")

#binary_crop = binary[880:3200,:]


binary_crop = maskBGR[500:2200,500:1800]

## Morphology for closing small holes created by the backgroundseparation
#lightly
se = np.ones((7,7), dtype='uint8')
img_close_light = cv2.morphologyEx(binary_crop, cv2.MORPH_CLOSE, se)

# strongly 
se = np.ones((20,20), dtype='uint8')
img_close = cv2.morphologyEx(binary_crop, cv2.MORPH_CLOSE, se)

# closing completely herbivory for external leave contour
se = np.ones((100,100), dtype='uint8')
leaf_binary = cv2.morphologyEx(binary_crop, cv2.MORPH_CLOSE, se)

# number of pixels with holes for area estimation

leaf_pxl = np.count_nonzero(img_close)

# counting area without holes:

leaf_whole = np.count_nonzero(leaf_binary)

### percentage

estimate = np.round(leaf_pxl / leaf_whole * 100,2)

####
