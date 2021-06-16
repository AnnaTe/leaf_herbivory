#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 10:37:33 2021

@author: Anna
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


# remove light colors from original image (background)
minBGR = np.array([0,0,0])
maxBGR = np.array([100, 200,150])
maskBGR = cv2.inRange(img, minBGR, maxBGR)
seg = cv2.bitwise_and(img, img, mask=maskBGR)


nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(maskBGR, connectivity=8)
sizes = stats[1:, -1]
nb_components = nb_components - 1
mask = np.zeros(maskBGR.shape, dtype='uint8')
for i in range(0, nb_components):
    if sizes[i] >= 1:
        mask[output == i + 1] = 255

blob = cv2.bitwise_and(seg, seg, mask=output)

from matplotlib.patches import Circle
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, figsize=(15, 10))
ax.set_aspect('equal')
number, output, stats, centroids = cv2.connectedComponentsWithStats(self.blob[:, :, 0], connectivity=8)
center = list(zip(centroids[1:, 0].astype(int), centroids[1:, 1].astype(int)))
radius = stats[1:, 3]

ax.imshow(cv2.cvtColor(self.cropped, cv2.COLOR_BGR2RGB))
ax.axis("off")

for i in range(centroids[1:, 1].shape[0]):
    circ = Circle(center[i], radius[i], color="r", linewidth=0.5, fill=False)
    ax.add_patch(circ)

# Show the image
return plt.show()

