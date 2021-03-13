# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 06:43:34 2019

@author: E442282
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

def applyRoberts(gray):
    
    filterX = np.array([[0,1],[-1,0]])
    filterY = np.array([[1,0],[0,-1]])
    img_X = cv2.filter2D(gray, -1, filterX)
    img_Y = cv2.filter2D(gray, -1, filterY)
    
    return img_X+img_Y

def applySobel(gray):
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    
    return abs_sobelx+abs_sobely


img = cv2.imread(r'C:/SAI/IIIT/2019_Monsoon/DIP/Project/input/1.jpg')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#edges = cv2.Canny(gray,50,150,apertureSize = 3)
edges = applyRoberts(gray)




#edges = cv2.Canny(gray,50,150,apertureSize = 3)
minLineLength = 1
maxLineGap = 10
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)


plt.figure(figsize=(12, 12))
plt.axis('off')
plt.imshow(img)
plt.show()
