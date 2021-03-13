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

def applySobel(gray):
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    
    return abs_sobelx+abs_sobely


images_path=r'C:\SAI\IIIT\2019_Monsoon\DIP\Project\input'
images=os.listdir(images_path)

output_path=r'C:\SAI\IIIT\2019_Monsoon\DIP\Project\output_hsv_mask'

#for im in images[:]:
#    
#    img = cv2.imread(os.path.join(images_path,im))
#        
#    adjusted=getHistogramAdjusted(img)
#    bilateral = cv2.bilateralFilter(img, 7, sigmaSpace = 75, sigmaColor =75)
#    hsv_mask=getHSVMask(bilateral)
#    
#    _,gray=getColorSpaces(bilateral)
#    
#    _,gray2=getColorSpaces(hsv_mask)
#    
#    imgs_comb = np.hstack([img,hsv_mask])   
#    plt.figure(figsize=(12, 12))
#    plt.axis('off')
#    plt.title(im)
#    plt.imshow(imgs_comb,cmap='gray')   

for im in images[:]:
    
    img = cv2.imread(os.path.join(images_path,im))
    image=getHistogramAdjusted(img)
    
    hMin = 0
    sMin = 0
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
    output = cv2.bitwise_and(image,image, mask= mask1)
    
    _,gray=getColorSpaces(output)
        
#    ret_thresh,im_bw = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(im_bw,kernel,iterations =1)
    
    init_mask=erosion.copy()
    mask = np.zeros(image.shape[:2],np.uint8)
    mask[init_mask == 255] = 1
    mask[init_mask == 0] = 2 #Guess everything else is background
    
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    
    mask, bgdModel, fgdModel = cv2.grabCut(image,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
    
    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    mask[mask == 1] = 255

#    _,gray=getColorSpaces(output)
#    _,contours,h = cv2.findContours(gray,1,2)
#    
#    for cnt in contours:
#        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
##        print( len(approx))
#        if len(approx)==5:
##            print( "pentagon")
#            cv2.drawContours(output,[cnt],0,(0,0,255),-1)
#        elif len(approx)==3:
##            print ("triangle")
#            cv2.drawContours(output,[cnt],0,(0,0,255),-1)
#        elif len(approx)==4:
##            print( "rectangle/square")
#            cv2.drawContours(output,[cnt],0,(0,0,255),-1)
#        elif len(approx) == 9:
##            print( "half-circle")
#            cv2.drawContours(output,[cnt],0,(0,0,255),-1)
#        elif len(approx) > 15:
##            print( "circle")
#            cv2.drawContours(output,[cnt],0,(0,0,255),-1)

#    imgs_comb = np.hstack([image,output])   
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.title(im)
    plt.imshow(mask,cmap='gray')    
    

    
#images_path=r'C:\SAI\IIIT\2019_Monsoon\DIP\Project\input_hsv_mask'
#images=os.listdir(images_path)
#
#output_path=r'C:\SAI\IIIT\2019_Monsoon\DIP\Project\output_hsv_mask'
#
#for im in images[:]:
#    print(im)
#    img = cv2.imread(os.path.join(images_path,im))
#        
#    adjusted=getHistogramAdjusted(img)
#    bilateral = cv2.bilateralFilter(adjusted, 7, sigmaSpace = 75, sigmaColor =75)
#    
#    rgb,gray=getColorSpaces(bilateral)
#    canny=CannyThreshold(100,gray,img)
 
#    plt.axis('off')
#    plt.title('Bilateral')
#    plt.imshow(bilateral,cmap='gray')
#    plt.show()
#    
#    plt.axis('off')
#    plt.title('Binary after Bilateral')
#    plt.imshow(binary,cmap='gray')
#    plt.show()
#    
#    plt.axis('off')
#    plt.title('HSV Mask-Trial and Error')
#    plt.imshow(hsv_mask,cmap='gray')
#    plt.show()
#    
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
    
#    imgs_comb = np.hstack([img,canny])
#    plt.axis('off')
#    plt.title('Canny')
#    plt.imshow(imgs_comb,cmap='gray')
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
##    plt.axis('off')
##    plt.title('Foreground-Stripes')
##    plt.imshow(mask,cmap='gray')
##    plt.show()
##    
##    cv2.imwrite(os.path.join(output_path,im), mask)
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
#    plt.axis('off')
#    plt.title('Foreground-Stripes')
#    plt.imshow(img,cmap='gray')
#    plt.show()
    

#plt.figure(figsize=(12, 12))
#gray = cv2.imread('C:/SAI/IIIT/2019_Monsoon/DIP/Project/zebra.jpg',0)
#    
#f = np.fft.fft2(gray)
#fshift = np.fft.fftshift(f)
#magnitude_spectrum = 20*np.log(np.abs(fshift))
#
#imgs_comb = np.hstack([gray,magnitude_spectrum])
#
##plt.subplot(2,3,2)
#plt.axis('off')
#plt.title('magnitude_spectrum')
#plt.imshow(imgs_comb,cmap='gray')   
#
#plt.show()



















