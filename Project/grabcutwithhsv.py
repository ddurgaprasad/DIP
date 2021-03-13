# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 19:43:18 2019

@author: E442282
"""


import numpy as np
import cv2 

import sys
from matplotlib import pyplot as plt


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



def getHSVMask(im):   
    hMin = 0
    sMin = 0
    vMin = 220
    
    hMax = 180
    sMax = 20
    vMax = 255
    
    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])
    
    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, lower, upper)

    return hsv_mask

img = cv2.imread('output_49.jpg')

adjusted=getHistogramAdjusted(img)
bilateral = cv2.bilateralFilter(adjusted, 7, sigmaSpace = 75, sigmaColor =75)




#rgb,gray=getColorSpaces(bilateral)
#mask= getBinaryImage(gray,220)

hsv_mask=getHSVMask(bilateral)
mask= hsv_mask.copy()

plt.axis('off')
plt.imshow(img,cmap='gray')
plt.show()

plt.axis('off')
plt.imshow(mask,cmap='gray')
plt.show()

img2 = img.copy()                             

output = np.zeros(img.shape, np.uint8)   
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

#https://stackoverflow.com/questions/53887425/opencv-grabcut-doesnt-update-mask-when-on-gc-init-with-mask-mode

init_mask = mask.copy()

mask[init_mask == 255] = 1
mask[init_mask == 0] = 2 #Guess everything else is background

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

mask, bgdModel, fgdModel = cv2.grabCut(img2,mask,None,bgdModel,fgdModel,2,cv2.GC_INIT_WITH_MASK)

mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
mask[mask == 1] = 255
plt.axis('off')
plt.imshow(mask,cmap='gray')
plt.show()
