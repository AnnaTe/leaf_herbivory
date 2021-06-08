import cv2
import numpy as np


class Data:
    """This is the Data class that stores the images and the altered copys for fast accessibility."""

    def __init__(self, path):
        self.img = self._open(path)
        self.cropped = np.copy(self.img)
        self.seg = np.copy(self.cropped)
        self.blob = np.copy(self.cropped)
        self.path = path
        self.name = self.path.split('/')[-1].split('.')[0]

    @staticmethod
    def _open(path):
        img = cv2.imread(path)
        return img

    def crop(self, percent=50):
        """crop of image from the center in percent"""
        if percent == 100:
            pass
        # if percent == 0:
        lower = (50 - (percent / 2)) / 100
        upper = (50 + (percent / 2)) / 100
        self.cropped = self.img[int(self.img.shape[0] * lower):int(self.img.shape[0] * upper) + 1,
                       int(self.img.shape[1] * lower):int(self.img.shape[1] * upper) + 1, :]
    
    def green(self):
        """segmentation of green area in the image with minimum size of segmented Components"""
        minBGR = np.array((0, 80, 0))
        maxBGR = np.array((100, 255, 100))
        maskBGR = cv2.inRange(self.img, minBGR, maxBGR)
        self.seg = cv2.bitwise_and(self.img, self.img, mask=maskBGR)

        #self.blob = self.seg
            

    def yellow(self, image, lowsize=100):
        """segmentation of yellow area in the image with minimum size of segmented Components"""
        minBGR = np.array((0, 133, 200))
        maxBGR = np.array((122, 255, 255))
        maskBGR = cv2.inRange(image, minBGR, maxBGR)
        self.seg = cv2.bitwise_and(image, image, mask=maskBGR)

        if lowsize == 0:
            self.blob = self.seg

        else:
            # blob dectection including sizes
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(maskBGR, connectivity=8)
            sizes = stats[1:, -1]
            nb_components = nb_components - 1
            mask = np.zeros(maskBGR.shape, dtype='uint8')
            for i in range(0, nb_components):
                if sizes[i] >= lowsize:
                    mask[output == i + 1] = 255
            self.blob = cv2.bitwise_and(image, image, mask=mask)
            return self.blob

    def filter(self, percent=50, lowsize=100):
        """calls the crop and segmentation methods together"""
        self.crop(percent)
        self.yellow(self.cropped, lowsize)
        
    def edgedetect(self, degree):
        self.blob 

    def circleplot(self):
        """plots red circles around the flowers. Is only called directly from Data class."""
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


#pathtest = "/home/rio/Dokumente/Uni/Master/Module/Fieldecology/images/leaf.jpeg"
pathtest = "/home/rio/Dokumente/Uni/Master/Module/Fieldecology/images/leaf_ideal.jpeg"

test = Data(pathtest)

test.img
test.green()

import matplotlib.pyplot as plt

plt.imshow(test.img)




plt.imshow(test.seg)

################## Canny Edge detection ###############

import cv2 as cv
import argparse
max_lowThreshold = 100
window_name = 'Edge Map'
title_trackbar = 'Min Threshold:'
ratio = 3
kernel_size = 3
def CannyThreshold(val):
    low_threshold = val
    img_blur = cv.blur(src_gray, (3,3))
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    mask = detected_edges != 0
    dst = src * (mask[:,:,None].astype(src.dtype))
    cv.imshow(window_name, dst)
parser = argparse.ArgumentParser(description='Code for Canny Edge Detector tutorial.')
parser.add_argument('--input', help='Path to input image.', default='/home/rio/Dokumente/Uni/Master/Module/Fieldecology/images/leaf_ideal.jpeg')
args = parser.parse_args()
src = cv.imread(cv.samples.findFile(args.input))
if src is None:
    print('Could not open or find the image: ', args.input)
    exit(0)
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
cv.namedWindow(window_name)
cv.createTrackbar(title_trackbar, window_name , 0, max_lowThreshold, CannyThreshold)
CannyThreshold(0)
cv.waitKey()

#################### find contours #####

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
src = cv.imread("/home/rio/Dokumente/Uni/Master/Module/Fieldecology/images/leaf_ideal.jpeg")
#if src is None:
#    print('Could not open or find the image:', args.input)
#    exit(0)
# Convert image to gray and blur it
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3,3))
# Create Window
#source_window = 'Source'
#cv.namedWindow(source_window)
cv.imshow(src)
max_thresh = 255
thresh = 100 # initial threshold
#cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)
#cv.waitKey()

########## unpack
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

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

