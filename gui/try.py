#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 10:36:28 2021

@author: rio
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

os.getcwd()
os.chdir("/home/rio/Dokumente/Uni/Master/Module/Fieldecology/leaf_herbivory/images")

#pathtest = "raw/IMG_4735.jpg"
#img = cv2.imread(pathtest)
img = cv2.imread("raw/IMG_4769.jpg")
img = img[500:2200,500:1800]

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# remove non green areas as HLS colortype
green_min = np.array([5, 50, 50],np.uint8)
green_max = np.array([15, 255, 255],np.uint8)

hls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
mask = cv2.inRange(hls, (30,20, 60), (75, 180, 255))
crop = cv2.bitwise_and(img, img, mask=mask)

################ not neccessary #######
# grey image

img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img_grey, "Greys_r")

crop_grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

# remove light colors to extract background and holes
black = np.array([0])
lightest_grey = np.array([110])
maskgrey = cv2.inRange(img_grey,black,lightest_grey)
crop_bw = cv2.bitwise_and(img_grey, img_grey, mask=maskgrey)


#histogramm equilisation
## too strong
img_adj = cv2.equalizeHist(img_grey)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply( cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

plt.imshow(cl1, "Greys_r")

seg = cv2.bitwise_and(cl1, cl1, mask=maskBGR)


##############remove later


## Morphology for closing small holes created by the backgroundseparation
#lightly
se = np.ones((7,7), dtype='uint8')
img_close_light = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, se)

# strongly 
se = np.ones((20,20), dtype='uint8')
img_close = cv2.morphologyEx(binary_crop, cv2.MORPH_CLOSE, se)

# closing completely herbivory for external leave contour
se = np.ones((100,100), dtype='uint8')
leaf_binary = cv2.morphologyEx(binary_crop, cv2.MORPH_CLOSE, se)

# number of pixels with holes for area estimation

leaf_pxl = np.count_nonzero(img_close_light)

# counting area without holes:

leaf_whole = np.count_nonzero(leaf_binary)

### percentage

estimate = np.round(leaf_pxl / leaf_whole * 100,2)


###### because of chunk on the side we need the convex hull 

#### 
#threshold = 10

plt.imshow(canny_output, "Greys")

canny_output = cv2.Canny(leaf_binary, 0,255)
# find contours
contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

area = []
for x in contours:
    area.append(cv2.contourArea(x))

hull = cv2.convexHull(contours[np.argmax(area)])

hull_list = []
for i in range(len(contours)):
    hull = cv2.convexHull(contours[i])
    hull_list.append(hull)


area = []
for x in hull_list:
    area.append(cv2.contourArea(x))
