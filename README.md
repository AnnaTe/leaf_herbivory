# Leaf herbivory tool

This programm should calculated the percentage of area missing due to Herbivory in leaf images using python with the OpenCV (cv2) library. It was created as part of the Module Fieldecology at the University Freiburg.

## What the software provides:

* a graphical user interface with 
  * an area for plotting and exploring the implemented filter on to one image
  * the possibility to provide an directory with images that should all be processed
* calculation of the herbivory damage percentage
* images of the steps of the process

## How the software filter work:

* read image with OpenCV in BGR Colorspace
* detect and if neccessary adjust brightness (in HSL Colorspace) of image and just of green layer
* create mask from segmentation of all green area between HSL: (30, 20, 60) and (75, 180, 255)
* adjust mask and smooth output with morphological function
* count number of pixel for leaf with herbivory
* fill holes with stronger morphological function
* find contour and convex hull 
* calculate area of convex hull 
* calculate percentage of herbivory damage

## requirements of the images:

* leaf photos have to be taken with a white background
* avoid direct sun light and any strong light reflection on the leaf
* if possible cover or remove the leaf stems

### contact information:

* Anna Tenberg: Anna.Tenberg@saturn.uni-freiburg.de

