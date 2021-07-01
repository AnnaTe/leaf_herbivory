import cv2
import numpy as np


class Data:
    """This is the Data class that stores the images and the altered copys for fast accessibility."""
    def __init__(self, path):
        self.img = self._open(path)
        self.cropped = np.copy(self.img)
        self.seg = np.copy(self.img)
        self.erosion = np.copy(self.img)
        self.mask = np.copy(self.img)
        self.filled = np.copy(self.img)
        self.contours = np.copy(self.img)
        self.path = path
        self.name = self.path.split('/')[-1].split('.')[0]

    @staticmethod
    def _open(path):
        img = cv2.imread(path)
        return img
    
    def green(self):
     
        """segmentation of green area in the image """
        green_mean = np.mean(self.img[:,:,1])
        
        hls = cv2.cvtColor(self.img,cv2.COLOR_BGR2HLS)
        
        if green_mean < 150:
            img_g = cv2.convertScaleAbs(self.img[:,:,1], alpha=1.1, beta=1.7)
            img_a = cv2.merge((self.img[:,:,0], img_g, self.img[:,:,2]))
            hls = cv2.cvtColor(img_a,cv2.COLOR_BGR2HLS)
        
        ##brightness analysis
        brightness = np.mean(hls[:,:,1])
        if brightness < 100: 
            bgr = cv2.convertScaleAbs(self.img, alpha=1.1, beta=1.7)
            hls = cv2.cvtColor(bgr,cv2.COLOR_BGR2HLS)

        self.mask = cv2.inRange(hls, (30, 20, 60), (75, 180, 255))
        self.seg = cv2.bitwise_and(self.img, self.img, mask=self.mask)
        return self.seg
    
    def herbivory(self, correction = 2):
        dim = self.mask.shape
        
        # make kernel with original size
        smooth_kernel = np.ones((int(dim[0]/200),int(dim[1]/200)), dtype='uint8')
        fill_kernel = np.ones((int(dim[0]/10),int(dim[1]/10)), dtype='uint8')
        
        # adding white borders
        new_mask = np.zeros((dim[0]+1000,dim[1]+1000), self.mask.dtype)
        new_mask[500:dim[0]+500,500:dim[1]+500] = self.mask
        
        smooth = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, smooth_kernel)
        
        eros_kernel = np.ones((5,5),np.uint8)
        self.erosion = cv2.erode(smooth,eros_kernel,iterations = 1)
        
        leaf_pxl = np.count_nonzero(smooth)
        
        self.filled = cv2.morphologyEx(smooth, cv2.MORPH_CLOSE, fill_kernel)
           
        
        # canny edge detection
        #canny_output = cv2.Canny(self.filled, 0,255)
        
        # find contours
        contours, hierarchy = cv2.findContours(self.filled, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                   
        area_contour = []
        hull_list = []
        area_hull = []
        for i in range(len(contours)):
            area_contour.append(cv2.contourArea(contours[i]))
            hull = cv2.convexHull(contours[i])
            hull_list.append(hull)
            area_hull.append(cv2.contourArea(hull))
        
        self.contours = cv2.cvtColor(smooth, cv2.COLOR_GRAY2BGR)
        for i in range(len(contours)):
            cv2.drawContours(self.contours, contours, i,(0,255,0) , 15, cv2.LINE_8, hierarchy, 0)
            cv2.drawContours(self.contours, hull_list, i,(255,0,255) , 20)
        
        #con_perc = 100-leaf_pxl/max(area_contour)*100
        
        hull_perc = 100-leaf_pxl/max(area_hull)*100

        #fill_perc = 100-leaf_pxl/np.count_nonzero(self.filled)*100
        
        herb_perc = hull_perc - correction
        
        return herb_perc
        
    ###########################################################################
    
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

    def crop(self, percent=50):
        """crop of image from the center in percent"""
        if percent == 100:
            pass
        # if percent == 0:
        lower = (50 - (percent / 2)) / 100
        upper = (50 + (percent / 2)) / 100
        self.cropped = self.img[int(self.img.shape[0] * lower):int(self.img.shape[0] * upper) + 1,
                       int(self.img.shape[1] * lower):int(self.img.shape[1] * upper) + 1, :]
    
    
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




