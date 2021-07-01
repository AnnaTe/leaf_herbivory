#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 12:14:03 2021

@author: anna
"""
import os 

import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

"""
filter tests on one image for developement using openCV
"""
os.getcwd()
os.chdir("/home/rio/Dokumente/Uni/Master/Module/Fieldecology/leaf_herbivory")

pathtest = "images/leaf_ideal.png"
img = cv2.imread(pathtest)
plt.imshow(img)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img_gray, "Greys")
#src_gray = cv2.blur(img_gray, (3,3))
#plt.imshow(src_gray, "Greys")

#detect edge using canny
threshold = 150
canny_output = cv2.Canny(src_gray, threshold, threshold * 2)
plt.imshow(canny_output, "Greys")

# find contours
contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

## Calculation of area

area = []
for x in contours:
    area.append(cv2.contourArea(x))

hull = cv2.convexHull(contours[np.argmax(area)])

hull_list = []
for i in range(len(contours)):
    hull = cv2.convexHull(contours[i])
    hull_list.append(hull)

#drawing

drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
for i in range(len(contours)):
    color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
    cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
    cv2.drawContours(drawing, hull_list, i, color)
# 
plt.imshow(drawing)


#################


## here in extra window with adjustable threshold

import cv2 as cv
import numpy as np
import argparse
import random as rng

rng.seed(12345)
def thresh_callback(val):
    threshold = val
    # Detect edges using Canny
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    # Find contours
    contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Draw contours
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
    # Show in a window
    cv.imshow('Contours', drawing)
# Load source image
#parser = argparse.ArgumentParser(description='Code for Finding contours in your image tutorial.')
#parser.add_argument('--input', help='Path to input image.', default='HappyFish.jpg')
#args = parser.parse_args()
src = cv.imread("images/leaf_ideal.jpeg")
# Convert image to gray and blur it
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3,3))
# Create Window
source_window = 'Source'
cv.namedWindow(source_window)
cv.imshow(source_window, src)
max_thresh = 255
thresh = 100 # initial threshold
cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)
cv.waitKey()

