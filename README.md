# Leaf herbivory tool

This programm should calculated the percentage of area missing due to herbivory in leaf images using python with the OpenCV (cv2) library.

For the detection of the original outline of the leave and the detection of holes missing roughly the same functions should be applied using different inputs.

## Ideas:

* adjustment of histogramm (cv2.equalizeHist(image) ) [ ]
* blurring and threshold binary to get clean edge (gaussian, bitwise_and, ??) [ ]
* color extraction of green range [x]
* canny edge detection and find contour for visualisation [x]
* calculate pixel percentage with connectedComponentswithStats [ ]

### for large missing pieces on the edge:

* convex hull (evtl with  deficit calculation) [ ]

### try out:

* contour area from OpenCV [ ]
