#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 12:14:03 2021

@author: anna
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

"""
filter tests on one image for developement using openCV
"""


pathtest = "/home/rio/Dokumente/Uni/Master/Module/Fieldecology/images/leaf_ideal.jpeg"
img = cv2.imread(pathtest)
plt.imshow(img)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img_gray, "Greys")
src_gray = cv2.blur(img_gray, (3,3))
plt.imshow(src_gray, "Greys")

#detect edge using canny
threshold = 100
canny_output = cv2.Canny(src_gray, threshold, threshold * 2)
plt.imshow(canny_output, "Greys")

# find contours
contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
for i in range(len(contours)):
    color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
    cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
# 
plt.imshow(drawing)