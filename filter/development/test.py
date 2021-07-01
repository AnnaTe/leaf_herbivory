#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 17:09:23 2021

@author: rio
"""

import os
import cv2

os.getcwd()

img = cv2.imread("../../pxl.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_mask = cv2.imread("../../pxl_mask.png")

img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)

np.count_nonzero(img)/np.count_nonzero(img_mask)

## test -> 73 %