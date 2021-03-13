# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 09:21:53 2019

@author: E442282
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 19:43:18 2019

@author: E442282
"""


import numpy as np
import cv2 
import os
import sys
from matplotlib import pyplot as plt
from skimage.transform import pyramid_gaussian

def getColorSpaces(image):
    rgb = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)    
    return rgb,gray

def getImageDimnesion(image):
    height,width = image.shape[:2]    
    return height,width

def showImage(image,title,cmap):
    plt.imshow(image,cmap=cmap)
    plt.axis('off')
    plt.title(title)


def splitRGBChannels(image):
  red, green, blue= cv2.split(image)  
  return red, green, blue
                               
def getBinaryImage(gray,thr=127):
    ret,thresh= cv2.threshold(gray,thr,255,cv2.THRESH_BINARY)
    return thresh
        
def getHistogramAdjusted(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)    
    lab_planes = cv2.split(lab)    
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))    
    lab_planes[0] = clahe.apply(lab_planes[0])    
    lab = cv2.merge(lab_planes)    
    adjusted = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)    
    return adjusted   

def applySobel(gray):
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    
    return abs_sobelx+abs_sobely

def getScharr(gray):
    # compute the Scharr gradient magnitude representation of the images
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
     
    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    
    return gradient

def getPyramids(image):
    height,width= getImageDimnesion(image)
    layer=image.copy()
    pyramids=[]
    for i in range(8):
        layer = cv2.pyrDown(layer) 
        layer=cv2.resize(layer,(width,height))
        pyramids.append(layer)
        
    return  pyramids   

def getLaplacianPyramids(image):
    height,width= getImageDimnesion(image)
    layer=image.copy()
    pyramids=[]
    for i in range(8):
        layer = cv2.pyrDown(layer) 
        up = cv2.pyrUp(layer)
        
        print(layer.shape,up.shape)
#        up = np.zeros(image.shape[:2])
        
#        lap = layer - up
        layer=cv2.resize(lap,(width,height))
        pyramids.append(layer)
        
    return  pyramids 
    


images_path=r'C:\SAI\IIIT\2019_Monsoon\DIP\Project\input'


image = cv2.imread(r'C:\SAI\IIIT\2019_Monsoon\DIP\Project\input\7.jpg')
#plt.figure(figsize=(12, 12))
#pyramids=getPyramids(image)
#
#getLaplacianPyramids(image)
#
#imgs_comb = np.hstack(pyramids)
#plt.axis('off')
#plt.title('pyramids')
#plt.imshow(imgs_comb,cmap='gray')
#plt.show()
 

NumLevels=8 
lower = image.copy()
 # Create a Gaussian Pyramid
gaussian_pyr = [lower]
for i in range(NumLevels):
    lower = cv2.pyrDown(lower)
    gaussian_pyr.append(lower)      
        
# Last level of Gaussian remains same in Laplacian
laplacian_top = gaussian_pyr[-1]
 
# Create a Laplacian Pyramid
laplacian_pyr = [laplacian_top]
for i in range(NumLevels,0,-1):
    size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
    gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize=size)
    laplacian = cv2.subtract(gaussian_pyr[i-1], gaussian_expanded)
    laplacian_pyr.append(laplacian) 

height,width= getImageDimnesion(image)
laplacian_pyr_resized=[]
laplacian_pyr_resized_sobel=[]
for lap in laplacian_pyr:    
    resized=cv2.resize(lap,(width,height))
    laplacian_pyr_resized.append(resized)
    
    _,gray =getColorSpaces(resized)
    binary= applySobel(gray)
    
    laplacian_pyr_resized_sobel.append(binary)
    
#    plt.axis('off')
#    plt.title('Laplacian Pyramids')
#    plt.imshow(binary,cmap='gray')
#    plt.show()     
    
 
print('=======================================')    

gaussian_pyr_resized=[]
gaussian_pyr_resized_sobel=[]
for gaussian in gaussian_pyr:    
    resized=cv2.resize(gaussian,(width,height))
    gaussian_pyr_resized.append(resized)    
    
    _,gray =getColorSpaces(resized)
    binary= applySobel(gray)
    
    
    gaussian_pyr_resized_sobel.append(binary)
    
#    plt.axis('off')
#    plt.title('Gaussian Pyramids')
#    plt.imshow(binary,cmap='gray')
#    plt.show()     
#    

plt.axis('off')
plt.title('Laplacian')
plt.imshow(laplacian_pyr_resized_sobel[7],cmap='gray')
plt.show()    

plt.axis('off')
plt.title('Gaussian')
plt.imshow(gaussian_pyr_resized_sobel[0],cmap='gray')
plt.show()

    
plt.axis('off')
plt.title('Image')
plt.imshow(image,cmap='gray')   
plt.show()
    
_,gray =getColorSpaces(image)


res=cv2.add(gray,laplacian_pyr_resized_sobel[7],dtype=cv2.CV_8U)

plt.axis('off')
plt.title('Image')
plt.imshow(res,cmap='gray')   
plt.show()
    















    