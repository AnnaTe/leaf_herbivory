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

ret,thresh1 = cv2.threshold(maskBGR,80,255,cv2.THRESH_BINARY)
plt.imshow(thresh1)

th3 = cv2.adaptiveThreshold(maskBGR,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

img = cv2.medianBlur(maskBGR,5)
ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()  
    

#detect edge using canny
threshold = 238
canny_output = cv2.Canny(th3, threshold, threshold * 2)
plt.imshow(canny_output, "Greys")



