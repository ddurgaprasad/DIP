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
#    hMin = 0
#    sMin = 0
#    vMin = 220
#    
#    hMax = 180
#    sMax = 80
#    vMax = 255
    
    hMin = 45
    sMin = 10
    vMin = 220
    
    hMax = 180
    sMax = 180
    vMax = 250
    
    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])
    
    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, lower, upper)

    return hsv_mask

def applyKMeansClustering(img):
    
    Z = img.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)    
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    
    return res2

    

images_path=r'C:\SAI\IIIT\2019_Monsoon\DIP\Project\input_hsv_mask'
images=os.listdir(images_path)

output_path=r'C:\SAI\IIIT\2019_Monsoon\DIP\Project\output_hsv_mask'

for im in images:
    print(im)
    img = cv2.imread(os.path.join(images_path,im))
    
    
    adjusted=getHistogramAdjusted(img)
    bilateral = cv2.bilateralFilter(adjusted, 7, sigmaSpace = 75, sigmaColor =75)
    
    rgb,gray=getColorSpaces(bilateral)
    binary= getBinaryImage(bilateral,220)
    
    
    
    hsv_mask=getHSVMask(binary)
    
#    plt.axis('off')
#    plt.title('Original')
#    plt.imshow(img,cmap='gray')
#    plt.show()
#    
#    plt.axis('off')
#    plt.title('Bilateral')
#    plt.imshow(bilateral,cmap='gray')
#    plt.show()
    
    plt.axis('off')
    plt.title('Binary after Bilateral')
    plt.imshow(binary,cmap='gray')
    plt.show()
    
#    plt.axis('off')
#    plt.title('HSV Mask-Trial and Error')
#    plt.imshow(hsv_mask,cmap='gray')
#    plt.show()
    
#    kernel = np.ones((5,5),np.uint8)
#    erosion = cv2.erode(hsv_mask,kernel,iterations = 1)
#    
#    opening = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel)
#    
#    plt.axis('off')
#    plt.title('Erosion')
#    plt.imshow(erosion,cmap='gray')
#    #plt.imshow(opening,cmap='gray')
#    plt.show()
#    
#    
#    #https://stackoverflow.com/questions/53887425/opencv-grabcut-doesnt-update-mask-when-on-gc-init-with-mask-mode
#    
#    init_mask=hsv_mask.copy()
#    mask = np.zeros(img.shape[:2],np.uint8)
#    mask[init_mask == 255] = 1
#    mask[init_mask == 0] = 2 #Guess everything else is background
#    
#    bgdModel = np.zeros((1,65),np.float64)
#    fgdModel = np.zeros((1,65),np.float64)
#    
#    mask, bgdModel, fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
#    
#    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
#    mask[mask == 1] = 255
#    
#    plt.axis('off')
#    plt.title('Foreground-Stripes')
#    plt.imshow(mask,cmap='gray')
#    plt.show()
#    
#    cv2.imwrite(os.path.join(output_path,im), mask)
#    _,contours,h = cv2.findContours(mask,1,2)
#    
#    for cnt in contours:
#        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
#        print( len(approx))
#        if len(approx)==5:
#            print( "pentagon")
#            cv2.drawContours(img,[cnt],0,255,-1)
#        elif len(approx)==3:
#            print ("triangle")
#            cv2.drawContours(img,[cnt],0,(0,255,0),-1)
#        elif len(approx)==4:
#            print( "rectangle/square")
#            cv2.drawContours(img,[cnt],0,(0,0,255),-1)
#        elif len(approx) == 9:
#            print( "half-circle")
#            cv2.drawContours(img,[cnt],0,(255,255,0),-1)
#        elif len(approx) > 15:
#            print( "circle")
#            cv2.drawContours(img,[cnt],0,(0,255,255),-1)
#    
#    cv2.imwrite(os.path.join(output_path,im), img)




