#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 17:18:31 2021

@author: rio
"""
####fehlerquote ohne frass, mitteln und von allen abziehen

new_image = np.zeros(im.img.shape, image.dtype)
alpha = 1.0 # Simple contrast control
beta = 0    # Simple brightness control
# Initialize values
new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# Do the operation new_image(i,j) = alpha*image(i,j) + beta
# Instead of these 'for' loops we could have used simply:
# new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
# but we wanted to show you how to access the pixels :)
for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        for c in range(image.shape[2]):
            new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)


r_image, g_image, b_image = cv2.split(im3.img)

r_image_eq = cv2.equalizeHist(r_image)
g_image_eq = cv2.equalizeHist(g_image)
b_image_eq = cv2.equalizeHist(b_image)

image_eq = cv2.merge((r_image_eq, g_image_eq, b_image_eq))

fig = plt.figure(figsize=(10, 20))

ax1 = fig.add_subplot(2, 2, 1)
ax1.axis("off")
ax1.title.set_text('Original')
ax2 = fig.add_subplot(2, 2, 2)
ax2.axis("off")
ax2.title.set_text("Equalized")

ax1.imshow(image_src, cmap=cmap_val)
ax2.imshow(image_eq, cmap=cmap_val)


    def subplots(self, image, title = "Title", subtitle, ax):
        
        import matplotlib.pyplot as plt
        
        axis = tuple(["ax" + str(i) for i in range(1, sp_nr+1)])
        
        fig , axis = plt.subplots(1,sp_nr)
        
        fig.suptitle(title, fontsize=20)
                    
        for i in range(len(axis)):
            axis[i].set_title(str(subtitles[i]))
            axis[i].imshow(image)
            axis[i].axis["off"]
        plt.tight_layout()
        # Make space for title
        plt.subplots_adjust(top=0.85)
        return plt.show()
