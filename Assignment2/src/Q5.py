# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 22:20:58 2019

@author: E442282
"""

import numpy as np
import cv2
import os,sys
from matplotlib import pyplot as plt
from numpy.lib.stride_tricks import as_strided


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
    blur = cv2.GaussianBlur(gray,(3,3),0)
    # Apply Laplacian operator in some higher datatype
#    Since our input is CV_8U we define ddepth = CV_16S to avoid overflow
#    laplacian = cv2.Laplacian(blur,cv2.CV_16S,ksize=3)
    # But this tends to localize the edge towards the brighter side.
#    laplacian1 = cv2.convertScaleAbs(laplacian)

    filter1 = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    laplacian1 = cv2.filter2D(blur, -1, filter1)
    laplacian1 = cv2.convertScaleAbs(laplacian1)

    return laplacian1

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
Sharpen the image
Use a gaussian smoothing filter and subtract the smoothed version from the original 
image (in a weighted way so the values of a constant area remain constant).
'''
def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    height,width=getImageDimnesion(image)
    low_contrast_mask=np.zeros((height,width), np.bool)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
    np.copyto(sharpened, image, where=low_contrast_mask)
    
    return sharpened   


# Low-pass kernel
    
def getNormalizedImage(image):
    norm_image=image.copy()
    norm_image = np.maximum(norm_image, np.zeros(norm_image.shape))
    norm_image = np.minimum(norm_image, 255 * np.ones(norm_image.shape))
    norm_image = norm_image.round().astype(np.uint8)
    
    return norm_image
  
import time
    
'''                 
Refered this material

https://jessicastringham.net/2017/12/31/stride-tricks/

'''
def applyFastFilter(gray, kernel):
    kernel = np.array(kernel)        
    view_shape = tuple(np.subtract(gray.shape, kernel.shape) + 1) + kernel.shape
    expanded_input  = as_strided(gray, shape = view_shape, strides = gray.strides * 2)  
    output=np.einsum('ij,ijkl->kl',kernel,expanded_input .T).T
    return output

    
def applyLowPass(gray,k,fastImplement=True):
    kernel = np.ones((k,k))/k*k
    padding=k//2
    if fastImplement:
        blur = applyFastFilter(gray,kernel)
        blur = np.pad(blur,(2,), 'constant')
    else:
        h,w = gray.shape
        blur = np.zeros(gray.shape)
        for i in range(padding,w-padding):
            for j in range(padding,h-padding):
                blur[i,j] = np.sum(kernel*gray[i-padding:i+padding +1,j-padding:j+padding + 1])
    
    return blur    



def applyFastFilterMedian(gray, kernel):
    view_shape = tuple(np.subtract(gray.shape, (k,k)) + 1) + (k,k)
    expanded_input = as_strided(gray, shape = view_shape, strides = gray.strides * 2)
    output=np.median(expanded_input,axis=(2,3))
    return output

    
def applyMedianFilter(gray,k,fastImplement=True):
    
    padding=k//2
    if fastImplement:
        filtered_gray = applyFastFilterMedian(gray,k)
        filtered_gray = np.pad(filtered_gray,(2,), 'constant')
    else:
        h,w = gray.shape
        filtered_gray = np.zeros(gray.shape)
        for i in range(padding,w-padding):
            for j in range(padding,h-padding):
                filtered_gray[i,j] = np.median(gray[i-padding:i+padding +1,j-padding:j+padding + 1])
    
    
    return filtered_gray



img = cv2.imread(r'C:\SAI\IIIT\2019_Monsoon\DIP\Assignment2\input_data\brain.jpg')
plt.figure(figsize=(12, 12))
rgb,gray=getColorSpaces(img)


kvals=[k for k in range(3,19,2)]
times_naive=[]
times_fast=[]

for k in range(3,19,2):
    
    start = time.time()
    #Get naive implementation time
    blurred_mage = applyMedianFilter(gray,k,False)    
    end = time.time()
    times_naive.append(end-start)

    start = time.time()
    #Get faster implementation(strides of numpy)
    blurred_mage = applyMedianFilter(gray,k,True)    
    end = time.time()
    times_fast.append(end-start)

plt.xlabel('K ')
plt.ylabel('Time taken')

plt.plot(kvals,times_naive)
plt.plot(kvals,times_fast)

plt.legend(['Naive','Fast'], loc='upper right')

plt.show()



#kernel_size=7
#blurred_mage1 = applyLowPass(gray,7,False)
#blurred_mage2 = applyLowPass(gray,7,True)
#plt.subplot(1,3,1) 
#plt.axis('off')
#plt.title('Original') 
#plt.imshow(gray,cmap='gray')
# 
#plt.subplot(1,3,2) 
#plt.axis('off')
#plt.title('Lowpass - Naive ')
#plt.imshow(blurred_mage1,cmap='gray')
#
#plt.subplot(1,3,3) 
#plt.axis('off')
#plt.title('Lowpass - Faster ')
#plt.imshow(blurred_mage2,cmap='gray')






#kvals=[k for k in range(3,19,2)]
#times_naive=[]
#times_fast=[]
#
#for k in range(3,19,2):
#    
#    start = time.time()
#    #Get naive implementation time
#    blurred_mage = applyLowPass(gray,k,False)    
#    end = time.time()
#    times_naive.append(end-start)
#
#    start = time.time()
#    #Get faster implementation(strides of numpy)
#    blurred_mage = applyLowPass(gray,k,True)    
#    end = time.time()
#    times_fast.append(end-start)
#
#plt.xlabel('K ')
#plt.ylabel('Time taken')
#
#
#plt.plot(kvals,times_naive)
#plt.plot(kvals,times_fast)
#
#plt.legend(['Naive','Fast'], loc='upper right')
#
#plt.show()










