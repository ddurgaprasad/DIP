# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 19:43:18 2019

@author: E442282
"""


import numpy as np
import cv2 
import os
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

def applySobel(gray):
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    
    return abs_sobelx+abs_sobely


def getGrabcutResponse(img,hsv_eroded=1):
    
    image=getHistogramAdjusted(img)
    
    hMin = 0
    sMin = 0
#    vMin = 210
    vMin = 210
    
    hMax = 179
    sMax = 180
    vMax = 255

    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower, upper)   
    
    
    kernel = np.ones((5,5),np.uint8)
    hsv_erosion = cv2.erode(mask1,kernel,iterations =1)
    
    if hsv_eroded==0:
        output = cv2.bitwise_and(image,image, mask= mask1) 
    else:
        output = cv2.bitwise_and(image,image, mask= hsv_erosion) 

    _,gray=getColorSpaces(output)        
    ret_thresh,im_bw = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(im_bw,kernel,iterations =1)
    
    init_mask=erosion.copy()
    mask = np.zeros(image.shape[:2],np.uint8)
    mask[init_mask == 255] = 1
    mask[init_mask == 0] = 2 #Guess everything else is background
    
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    
    try:
        mask, bgdModel, fgdModel = cv2.grabCut(image,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
    except:
        pass
        
    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    mask[mask == 1] = 255
    
    
    return mask1,hsv_erosion,mask
    

#images_path=r'C:\SAI\IIIT\2019_Monsoon\DIP\Project\input'
#images=os.listdir(images_path)
#
#
#
#for im in images[:]:
#    
#    print(im)
#    img = cv2.imread(os.path.join(images_path,im))
#
#    hsv,hsv_erosion,grabcut_with_hsv_erosion = getGrabcutResponse(img,1)
#    _,_,grbacut_without_hsv_erosion = getGrabcutResponse(img,0)
#    
#    plt.figure(figsize=(12, 12))
#    plt.axis('off')
#    plt.title('hsv, hsv_erosion,grabcut_with_hsv_erosion,grbacut_without_hsv_erosion')
#    
#    plt.imshow(np.hstack((hsv, hsv_erosion,grabcut_with_hsv_erosion,grbacut_without_hsv_erosion)),cmap='gray')    
#    plt.show()
#
#


from PIL import Image

mask = Image.open( 'C:/SAI/IIIT/2019_Monsoon/DIP/Project/pytorch/FudanPed00001_mask.png')   


plt.figure(figsize=(12, 12))
plt.axis('off')
plt.title('mask penfudan')

plt.imshow(mask)    
plt.show()


#mask.putpalette([
#    0, 0, 0, # black background
#    255, 255, 255, # index 1 is red
#    255, 255, 255, # index 2 is yellow
#    255, 255, 255, # index 3 is orange
#])






mask1 = Image.open('C:/SAI/IIIT/2019_Monsoon/DIP/Project/pytorch/output_3.png')










    
    
    
    
    
    
    
    
    
    
    