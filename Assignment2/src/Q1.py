# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 22:20:58 2019

@author: E442282
"""

import numpy as np
import cv2
import os,sys
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
  red, green, blue= cv2.split(img)
  
  return red, green, blue
                               
def getHistogram(image, bins=256):
    
    image_pixels=image.flatten()
    # array with size of bins, set to zeros
    histogram = np.zeros(bins)
    
    # loop through pixels and sum up counts of pixels
    for pixel in image_pixels:
        histogram[pixel] += 1
    
    # return our final result
    return histogram

def applySobel(gray):
    
#    filterX = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
#    filterY= np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
#    
#    img_X = cv2.filter2D(gray, -1, filterX)
#    img_Y = cv2.filter2D(gray, -1, filterY)
#    
#    return img_X+img_Y

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    
    return abs_sobelx+abs_sobely

def applyRoberts(gray):
    
    filterX = np.array([[0,1],[-1,0]])
    filterY = np.array([[1,0],[0,-1]])
    img_X = cv2.filter2D(gray, -1, filterX)
    img_Y = cv2.filter2D(gray, -1, filterY)
    
    roberts=img_X+img_Y
    scale_factor = np.max(roberts)/255
    roberts = (roberts/scale_factor).astype(np.uint8)
    cv2.normalize(roberts, roberts, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    roberts = roberts.astype(np.uint8)
    
    return roberts

def applyPrewitt(gray):
    
    filterX = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    filterY= np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    
    img_X = cv2.filter2D(gray, -1, filterX)
    img_Y = cv2.filter2D(gray, -1, filterY)
    
    return img_X+img_Y

def applyLaplacian(gray):
    # Apply Gaussian Blur
    sigma=1.0
    kernel_size=3
    
    blur = cv2.GaussianBlur(gray,(kernel_size,kernel_size),sigma)
    # Apply Laplacian operator in some higher datatype
#    Since our input is CV_8U we define ddepth = CV_16S to avoid overflow
    laplacian = cv2.Laplacian(blur,cv2.CV_16S,ksize=3)
#     But this tends to localize the edge towards the brighter side.
#    laplacian1 = cv2.convertScaleAbs(laplacian)
    laplacian = np.maximum(laplacian, np.zeros(laplacian.shape))
    laplacian = np.minimum(laplacian, 255 * np.ones(laplacian.shape))
    laplacian = laplacian.round().astype(np.uint8)
    return laplacian

#    filter1 = np.array([[0,1,0],[1,-4,1],[0,1,0]])
#    laplacian1 = cv2.filter2D(blur, -1, filter1)
#    laplacian1 = cv2.convertScaleAbs(laplacian1)
#    return laplacian1

def applyCanny(gray,low):
    ratio=3 
    kernel_size = 3
    img_blur = cv2.blur(gray, (3,3))
    detected_edges = cv2.Canny(img_blur, low, low*ratio, kernel_size)
#    mask = detected_edges != 0
#    dst = gray * (mask[:,:,None].astype(gray.dtype))
    
    return detected_edges

def addGaussianNoise(gray):
    row,col= gray.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gaussian = np.random.normal(mean,sigma,(row,col))
    noisy_image = np.zeros(gray.shape, np.float32)
    noisy_image = gray + gaussian    
    cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image 
'''
 We set the low threshold to 0.66*[mean value] and set the high threshold to 1.33*[mean value]
'''
def getEdges_Mean(gray):
    min_threshold = 0.66 *np.mean(gray)
    max_threshold = 1.33 *np.mean(gray)
    edges = cv2.Canny(gray,min_threshold,max_threshold)    
    return edges
'''
 We set the low threshold to 0.66*[median value] and set the high threshold to 1.33*[median value]
'''
def getEdges_Median(gray):
    min_threshold = 0.66 *np.median(gray)
    max_threshold = 1.33 *np.median(gray)
    edges = cv2.Canny(gray,min_threshold,max_threshold)    
    return edges    

def getEdges_OTSU(gray):
    ret2,th2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(gray,(3,3),0)
    max_threshold=ret2
    min_threshold=0.5*max_threshold
    edges = cv2.Canny(blur,min_threshold,max_threshold)
    
    return edges



img = cv2.imread(r'C:\SAI\IIIT\2019_Monsoon\DIP\Assignment2\input_data\barbara.jpg')
plt.figure(figsize=(20, 20))
rgb,gray=getColorSpaces(img)

#e1 = cv2.getTickCount()
#roberts=applyRoberts(gray)
#
#
#e1 = cv2.getTickCount()
#sobel=applySobel(gray)
#e2 = cv2.getTickCount()
#time_taken = (e2 - e1)/ cv2.getTickFrequency()
#print(time_taken)
#
#e1 = cv2.getTickCount()
#prewitt=applyPrewitt(gray)
#e2 = cv2.getTickCount()
#time_taken = (e2 - e1)/ cv2.getTickFrequency()
#print(time_taken)
#
#e1 = cv2.getTickCount()
#laplacian=applyLaplacian(gray)
#e2 = cv2.getTickCount()
#time_taken = (e2 - e1)/ cv2.getTickFrequency()
#print(time_taken)
# 
#e1 = cv2.getTickCount()
#canny=applyCanny(gray,0)
#e2 = cv2.getTickCount()
#time_taken = (e2 - e1)/ cv2.getTickFrequency()
#print(time_taken)


#plt.subplot(3,2,1)
#plt.axis('off')
#plt.title('Original')
#plt.imshow(rgb)
#
#plt.subplot(3,2,2)
#plt.axis('off')
#plt.title('Roberts')
#plt.imshow(5*roberts,cmap='gray')

#plt.subplot(3,2,3)
#plt.axis('off')
#plt.title('Sobel')
#plt.imshow(sobel,cmap='gray')
#
#plt.subplot(3,2,4)
#plt.axis('off')
#plt.title('Prewitt')
#plt.imshow(prewitt,cmap='gray')
#
#plt.subplot(3,2,5)
#plt.axis('off')
#plt.title('Laplacian')
#plt.imshow(5*laplacian,cmap='gray')
#
#plt.subplot(3,2,6)
#plt.axis('off')
#plt.title('Canny')
#plt.imshow(canny,cmap='gray')
#
#plt.show()



#Roberts
#advantages
#    Very quick to compute.
#    Only 4 pixels are added/subtracted 
#    No Parameters
#Disadvanatages
#   Sensitive to noise -observe  at the arm    
#   Sharp edges are detected .. Weak edges are ignored 
#   Strongest edges like the table,book shelf are detected reliably 
# More sensitive to noise than sobel and others

#sharp intensity changes detect the edges very well
#Gradual changes are resulting weak/faint edges- Arm boundaries- both hands

#Sobel

#Slower compared to Roberts as 
#Less sensitive to noise as it has larger filter ->smooths input
#Less noise in the bacgroudn due to smoothing
#For the same edges  reponse is stronger in Sobel - notice the books,shelf
#and table legs  and left arm

#Edges are thicker than in original image due to smoothing effect of sobel

# Noise inroberts is also here but with less intensity by several orders
# This can be eliminated by thresholding


# After adding Gaussian noise also , sobel is detecting high frequency chnages

# Soomthly chanign surface intesity 

#noise rejection, edge detection and speed

# Prewitt is not isotorpic, sensitive to some directions
#is used for detecting vertical and horizontal edges in images
# Relatively inexpensive and threfore faster than sobel
#More Noise is present 


#Laplacian
# Single filter is used for both directions
#Caculates 2nd derivatives in single pass
#Therefore are very sensitive to noise
#This two-step process is called the Laplacian of Gaussian (LoG) operation.
#Hence to counter this Gaussian smoothing is applied before lapalcain egde detection
#Isotropic in nature- it produces a uniform edge magnitude for all directions.
#in areas where the image has a constant intensity (i.e. where the intensity gradient is zero), the LoG response will be zero.
#In the vicinity of a change in intensity, however, the LoG response will be positive on the darker side, and negative on the lighter side.
# Zero crossings are used to detect edges
#Silhoutte of face and hands is well approximately defined
#

#Canny
#All noise is removed
#The Gaussian kernel size, σ, also affects the edges detected. If σ is large, the more obvious, defining edges of the picture are retrieved. Conversely, 
#if σ is small, the finer edges are picked out as well.
#isotropic
#Removes noise based on σ value given 
# more number of parameters to tweak
#

gaussian_noise=gray.copy()
mean=0
stddev=8
cv2.randn(gaussian_noise, mean, stddev) 
noisy_gray=gray + gaussian_noise


roberts=applyRoberts(gray)
plt.subplot(5,2,1)
plt.axis('off')
plt.title('Roberts')
plt.imshow(5*roberts,cmap='gray')

roberts_noisy=applyRoberts(noisy_gray)
plt.subplot(5,2,2)
plt.axis('off')
plt.title('Roberts on Noisy ')
plt.imshow(5*roberts_noisy,cmap='gray')


sobel=applySobel(gray)
plt.subplot(5,2,3)
plt.axis('off')
plt.title('Sobel ')
plt.imshow(sobel,cmap='gray')

sobel_noisy=applySobel(noisy_gray)
plt.subplot(5,2,4)
plt.axis('off')
plt.title('Sobel on Noisy ')
plt.imshow(sobel_noisy,cmap='gray')

prewitt=applyPrewitt(gray)
plt.subplot(5,2,5)
plt.axis('off')
plt.title('Prewitt ')
plt.imshow(prewitt,cmap='gray')

prewitt_noisy=applyPrewitt(noisy_gray)
plt.subplot(5,2,6)
plt.axis('off')
plt.title('Prewitt on Noisy ')
plt.imshow(prewitt_noisy,cmap='gray')

laplacian=applyLaplacian(gray)
plt.subplot(5,2,7)
plt.axis('off')
plt.title('Laplacian ')
plt.imshow(laplacian,cmap='gray')

laplacian_noisy=applyLaplacian(noisy_gray)
plt.subplot(5,2,8)
plt.axis('off')
plt.title('Laplacian on Noisy ')
plt.imshow(laplacian_noisy,cmap='gray')

canny=applyCanny(gray,0)
plt.subplot(5,2,9)
plt.axis('off')
plt.title('Canny ')
plt.imshow(canny,cmap='gray')

canny_noisy=applyCanny(noisy_gray,0)
plt.subplot(5,2,10)
plt.axis('off')
plt.title('Laplacian on Noisy ')
plt.imshow(canny_noisy,cmap='gray')

plt.show()
#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

#Roberts
# More sensitive to noise than sobel and others
# Detecting additional points

#Sobel
#Larger convolutaionkernel ,less suciptible to noise
#Noise similar to Roberts but very less in intensity--can thresholded


#Prewitt
#Behaviur simialr to Sobel

#Laplacian
#The Laplacian is often applied to an image that has first been
# smoothed with something approximating a Gaussian smoothing filter in order to reduce its sensitivity to noise
# very sensitive to noise

#Canny 
#Noise is not not so impactful as in other cases

















