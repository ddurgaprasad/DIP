# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 19:43:18 2019

@author: E442282
"""


import numpy as np
import cv2 

from collections import Counter

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

def getImageArea(img):
    h,w=getImageDimnesion(img)
    return h*w
        

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

def raw_moment(data, i_order, j_order):
  nrows, ncols = data.shape
  y_indices, x_indicies = np.mgrid[:nrows, :ncols]
  return (data * x_indicies**i_order * y_indices**j_order).sum()


def moments_cov(data):
  data_sum = data.sum()
  m10 = raw_moment(data, 1, 0)
  m01 = raw_moment(data, 0, 1)
  x_centroid = m10 / data_sum
  y_centroid = m01 / data_sum
  u11 = (raw_moment(data, 1, 1) - x_centroid * m01) / data_sum
  u20 = (raw_moment(data, 2, 0) - x_centroid * m10) / data_sum
  u02 = (raw_moment(data, 0, 2) - y_centroid * m01) / data_sum
  cov = np.array([[u20, u11], [u11, u02]])
  return cov


img = cv2.imread('thr.jpg')

adjusted=getHistogramAdjusted(img)
bilateral = cv2.bilateralFilter(adjusted, 7, sigmaSpace = 75, sigmaColor =75)

#rgb,gray=getColorSpaces(bilateral)
#mask= getBinaryImage(gray,220)

hsv_mask=getHSVMask(bilateral)
mask= hsv_mask.copy()

#plt.axis('off')
#plt.imshow(img,cmap='gray')
#plt.show()
#
#plt.axis('off')
#plt.imshow(mask,cmap='gray')
#plt.show()

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
#plt.axis('off')
#plt.imshow(mask,cmap='gray')
#plt.show()


image, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#area = cv2.contourArea(cnt)

#cv2.drawContours(img, contours, -1, (0,255,0), 3)
#plt.axis('off')
#plt.imshow(img,cmap='gray')
#plt.show()


contours_new=[]
for cnt in contours:   
    if len(cnt) >=5:
        contours_new.append(cnt)

cv2.drawContours(img, contours_new, -1, (0,255,0), 3)
plt.axis('off')
plt.imshow(img,cmap='gray')
plt.show()
        

angles=[]

for cnt in contours_new:    
    rect = cv2.minAreaRect(cnt)
    angles.append(abs(int(rect[2])))
    
most_common,num_most_common = Counter(angles).most_common(1)[0] # 4, 6 times
print(most_common)


ellipse_angles=[]
for cnt in contours_new:    
    rect = cv2.minAreaRect(cnt)    
    if abs(int(rect[2]))==most_common :  
        ellipse = cv2.fitEllipse(cnt)
        ellipse_angles.append(int(ellipse[2]))

most_common_ellipse_angle,num_most_common = Counter(ellipse_angles).most_common(1)[0] # 4, 6 times
print(most_common_ellipse_angle)

       
    
for cnt in contours_new:    
    rect = cv2.minAreaRect(cnt)
    
    if abs(int(rect[2]))==most_common :  
#        print('Angle ',abs(int(rect[2])))        

        ellipse = cv2.fitEllipse(cnt)
#        print('Ellipse Angle ',int(ellipse[2]))
#        im=cv2.ellipse(img,ellipse,(255,255,0),2)
        if int(ellipse[2])==most_common_ellipse_angle : 
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            im = cv2.drawContours(img,[box],0,(0,0,255),2)
    

#plt.figure(figsize=(12, 12))
#plt.axis('off')
#plt.imshow(img)
#plt.show()












