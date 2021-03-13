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


def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    return labeled_img
    

images_path=r'C:\SAI\IIIT\2019_Monsoon\DIP\Project\detect_stripes_decent'
images=os.listdir(images_path)




for im in images[:]:
    
    print(im)
    
    img = cv2.imread(os.path.join(images_path,im))
    image=getHistogramAdjusted(img)

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    binary=getBinaryImage(gray)
    
    kernel = np.ones((5,5),np.uint8)
    binary = cv2.erode(binary,kernel,iterations =1)
#    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    ret, labels = cv2.connectedComponents(binary)
    output = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)

    
    components=imshow_components(labels)    
    new_gray = cv2.cvtColor(components,cv2.COLOR_BGR2GRAY)    
    new_binary=getBinaryImage(new_gray)
    
    _,contours,h = cv2.findContours(binary,1,2)
    

    rows,cols = image.shape[:2]
    for cnt in contours:

        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image,[box],0,(0,0,255),2)        
        
#        [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
#        lefty = int((-x*vy/vx) + y)
#        righty = int(((cols-x)*vy/vx)+y)
#        cv2.line(image,(cols-1,righty),(0,lefty),(0,255,0),2)

        
#    for cnt in contours:
#        rect = cv2.minAreaRect(cnt)
#        box = cv2.boxPoints(rect)
#        box = np.int0(box)
#        cv2.drawContours(image,[box],0,(0,0,255),2)
#        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
##        print( len(approx))
#        if len(approx)==5:
##            print( "pentagon")
#            cv2.drawContours(image,[cnt],0,(0,0,255),-1)
#        elif len(approx)==3:
##            print ("triangle")
#            cv2.drawContours(image,[cnt],0,(0,0,255),-1)
#        elif len(approx)==4:
##            print( "rectangle/square")
#            cv2.drawContours(image,[cnt],0,(0,0,255),-1)
#        elif len(approx) == 9:
##            print( "half-circle")
#            cv2.drawContours(image,[cnt],0,(0,0,255),-1)
#        elif len(approx) > 15:
##            print( "circle")
#            cv2.drawContours(image,[cnt],0,(0,0,255),-1)

    
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.title('Contour:'+im)
    plt.imshow(components)    
 
    
#
##From a matrix of pixels to a matrix of coordinates of non-black points.
##(note: mind the col/row order, pixels are accessed as [row, col]
##but when we draw, it's (x, y), so have to swap here or there)
#mat = np.argwhere(binary != 0)
#mat[:, [0, 1]] = mat[:, [1, 0]]
#mat = np.array(mat).astype(np.float32) #have to convert type for PCA
#
##mean (e. g. the geometrical center) 
##and eigenvectors (e. g. directions of principal components)
#m, e = cv2.PCACompute(mat, mean = np.array([]))
#
##now to draw: let's scale our primary axis by 100, 
##and the secondary by 50
#center = tuple(m[0])
#endpoint1 = tuple(m[0] + e[0]*100)
#endpoint2 = tuple(m[0] + e[1]*50)
#
#cv2.circle(img, center, 5, 255)
##    cv2.line(img, center, endpoint1, (255, 255, 0) ,2)
#cv2.line(img, center, endpoint2, (255, 255, 0),2)
#
#plt.axis('off')
#plt.title('PCA')
#plt.imshow(img  ,cmap='gray') 
#plt.show()













