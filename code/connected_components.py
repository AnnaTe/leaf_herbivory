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
os.chdir("/home/rio/Dokumente/Uni/Master/Module/Fieldecology/leaf_herbivory")

pathtest = "images/leaf_ideal.jpeg"
img = cv2.imread(pathtest)
plt.imshow(img)

# remove light colors from original image (background)
minBGR = np.array([0,0,0])
maxBGR = np.array([100, 200,150])
maskBGR = cv2.inRange(img, minBGR, maxBGR)
seg = cv2.bitwise_and(img, img, mask=maskBGR)


# grayscale image 

img_gray = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
plt.imshow(img_gray, "Greys")

# blur (optional)
src_gray = cv2.blur(img_gray, (3,3))
plt.imshow(src_gray, "Greys")


## treshold binary 

ret,thresh1 = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)


#detect edge using canny
threshold = 238
canny_output = cv2.Canny(maskBGR, threshold, threshold * 2)
plt.imshow(canny_output)






