import numpy as np
import cv2
import os,sys
from matplotlib import pyplot as plt


def getColorSpaces(image):
    rgb = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    
    return rgb,gray

def applyCustomFilter(gray):
    
#    filterX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])    
#    filterY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    gx = cv2.filter2D(gray, -1, filterX)
    gy = cv2.filter2D(gray, -1, filterY)
    #Calculate the gradient magnitude
    g = np.sqrt(gx * gx + gy * gy)
    #Normalize output to fit the range 0-255
    g *= 255.0 / np.max(g)
    
    return g,gx,gy

def getNormalizedImage(image):
    norm_image=image.copy()
    norm_image = np.maximum(norm_image, np.zeros(norm_image.shape))
    norm_image = np.minimum(norm_image, 255 * np.ones(norm_image.shape))
    norm_image = norm_image.round().astype(np.uint8)
    
    return norm_image
    
def applyCustomFilter2(gray,kernel):
    k=kernel.shape[0]
    padding=k//2

    w,h = gray.shape
    filtered = np.zeros(gray.shape)
    for i in range(padding,w-padding):
        for j in range(padding,h-padding):
            filtered[i,j] = np.sum(kernel*gray[i-padding:i+padding +1,j-padding:j+padding + 1])

    return filtered  


#img = cv2.imread(r'C:\SAI\IIIT\2019_Monsoon\DIP\Assignment2\input_data\box.png')
#plt.figure(figsize=(12, 12))
#rgb,gray=getColorSpaces(img)
#
#edges_all,edges_X,edges_Y=applyCustomFilter(gray)
#
#plt.subplot(3,2,1)
#plt.axis('off')
#plt.title('Original')
#plt.imshow(rgb)
#
#plt.subplot(3,2,2)
#plt.axis('off')
#plt.title('Edges')
#plt.imshow(edges_Y,cmap='gray')
#
#plt.show()


img = cv2.imread(r'C:\SAI\IIIT\2019_Monsoon\DIP\Assignment2\input_data\box.png')
plt.figure(figsize=(12, 12))
rgb,gray=getColorSpaces(img)


#filterX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  
#filterY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


filterX = np.array([[-2, 0, 2], 
                    [-2, 0, 2], 
                    [-2, 0, 2]])  
    
filterY = np.array([[-2, -2, -2], 
                    [0, 0, 0],
                    [2, 2, 2]])

horizontal_edge=applyCustomFilter2(gray,filterY)
horizontal_edge=getNormalizedImage(horizontal_edge)

vertical_edge=applyCustomFilter2(gray,filterX)
vertical_edge=getNormalizedImage(vertical_edge)

edges_all,_,_=applyCustomFilter(gray)
edges_all=getNormalizedImage(edges_all)

plt.subplot(141)
plt.axis('off')
plt.title('Original')
plt.imshow(rgb)

plt.subplot(142)
plt.axis('off')
plt.title('Horizontal Edges')
plt.imshow(horizontal_edge,cmap='gray')

plt.subplot(143)
plt.axis('off')
plt.title('Vetical Edges')
plt.imshow(vertical_edge,cmap='gray')

plt.subplot(144)
plt.axis('off')
plt.title('All edges')
plt.imshow(5*edges_all,cmap='gray')

plt.show()

































