#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 14:07:34 2021

@author: Anna
"""
import os
import cv2
import matplotlib.plyplot as plt
import numpy as np


imgtype = [img, mask, seg, filled, contours]


def subplots(img, imgtype, title = "Title", subtitles = imgtype):
    sp_nr = len(imgtype)
     

    axis = ["ax" + str(i) for i in range(1, sp_nr+1)]

    fig , tuple(axis) = plt.subplots(1,sp_nr)

    fig.suptitle(title, fontsize=20)
    
    for i in range(len(axis)):
        axis[i].set_title(subtitle[i])
        axis[i].imshow(imagetype[i])
        axis[i].axis["off"]
    plt.tight_layout()
# Make space for title
    plt.subplots_adjust(top=0.85)
    return plt.show()



fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,2, figsize (30,10))

fig.suptitle("Herbivory", fontsize = 20)

ax1.set_title("original")
ax1.imshow(cv2.cvtColor(im.img, cv2.COLOR_BGR2RGB))
ax1.axis("off")

ax2.set_title("mask")
ax2.imshow(im.mask, "Greys_r")
ax2.axis("off")

ax3.set_title("")
ax3.imshow()
ax3.axis("off")

ax4.set_title("filled")
ax4.imshow(im.filled, "Greys_r")
ax4.axis("off")

ax5.set_title("contour")
ax5.imshow(cv2.cvtColor(im.contours, cv2.COLOR_BGR2RGB))
ax5.axis("off")

plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()

    

dim = im.mask.shape

# make kernel with original size
smooth_kernel = np.ones((int(dim[0]/200),int(dim[1]/200)), dtype='uint8')
fill_kernel = np.ones((int(dim[0]/10),int(dim[1]/10)), dtype='uint8')

# adding white borders
new_mask = np.zeros((dim[0]+1000,dim[1]+1000), im.mask.dtype)
new_mask[500:dim[0]+500,500:dim[1]+500] = im.mask

smooth = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, smooth_kernel)

eros_kernel = np.ones((5,5),np.uint8)
im.erosion = cv2.erode(smooth,eros_kernel,iterations = 1)

leaf_pxl = np.count_nonzero(smooth)

im.filled = cv2.morphologyEx(smooth, cv2.MORPH_CLOSE, fill_kernel)
   

# canny edge detection
#canny_output = cv2.Canny(im.filled, 0,255)

# find contours
contours, hierarchy = cv2.findContours(im.filled, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
           
area_contour = []
hull_list = []
area_hull = []
for i in range(len(contours)):
    area_contour.append(cv2.contourArea(contours[i]))
    hull = cv2.convexHull(contours[i])
    hull_list.append(hull)
    area_hull.append(cv2.contourArea(hull))

im.contours = cv2.cvtColor(smooth, cv2.COLOR_GRAY2BGR)
for i in range(len(contours)):
    cv2.drawContours(im.contours, contours, i,(0,255,0) , 35, cv2.LINE_8, hierarchy, 0)
    cv2.drawContours(im.contours, hull_list, i,(255,0,255) ,40)

#con_perc = 100-leaf_pxl/max(area_contour)*100

hull_perc = 100-leaf_pxl/max(area_hull)*100

#fill_perc = 100-leaf_pxl/np.count_nonzero(im.filled)*100

herb_perc = hull_perc - 2

