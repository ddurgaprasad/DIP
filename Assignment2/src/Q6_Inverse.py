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

def getNormalizedImage(image):
    norm_image=image.copy()
    norm_image = np.maximum(norm_image, np.zeros(norm_image.shape))
    norm_image = np.minimum(norm_image, 255 * np.ones(norm_image.shape))
    norm_image = norm_image.round().astype(np.uint8)
    
    return norm_image

def splitRGBChannels(image):
  red, green, blue= cv2.split(img)
  
  return red, green, blue

def applyBilateralFilter(gray,sigma_domain,sigma_range):
    
    k=sigma_domain
    padding=k//2
    h,w = getImageDimnesion(gray) 
    view_shape = tuple(np.subtract(gray.shape, (k,k)) + 1) + (k,k)
    # Create a gausian filter    
    gaussX,gaussY = np.meshgrid(np.linspace(-(padding),(padding),k),np.linspace(-(padding),(padding),k))    
    kernel_domain = np.exp(-(gaussX**2 + gaussY**2)/(2*(sigma_domain**2)))
    kernel_domain=-kernel_domain
    
    expanded_input = as_strided(gray, shape = view_shape, strides = gray.strides * 2)
    kernel_range = expanded_input -expanded_input[:,:,padding,padding][:,:,np.newaxis,np.newaxis]    
    kernel_range = np.exp(-kernel_range/(2*sigma_range**2))
    kernel = kernel_range * kernel_domain
    
    filtered_gray = np.sum(kernel*expanded_input,axis=(2,3))/np.sum(kernel,axis=(2,3))
    return filtered_gray

img = cv2.imread(r'C:\SAI\IIIT\2019_Monsoon\DIP\Assignment2\input_data\ravva_upma.jpg')
plt.figure(figsize=(20, 20))
rgb,gray=getColorSpaces(img)

sigma_domain=15
sigma_range=1

sky_bilateral=applyBilateralFilter(gray,sigma_domain,sigma_range)


plt.subplot(231)
plt.axis('off')
plt.title('Original-Sky')
plt.imshow(gray,cmap='gray')

plt.subplot(232)
plt.axis('off')
plt.title('Bilateral-Sky')
plt.imshow(sky_bilateral,cmap='gray')


#img = cv2.imread(r'C:\SAI\IIIT\2019_Monsoon\DIP\Assignment2\input_data\noir.png')
#rgb,gray=getColorSpaces(img)
#noir_bilateral=applyBilateralFilter(gray,5,sigma_domain,sigma_range)
#
#plt.subplot(223)
#plt.axis('off')
#plt.title('Original-Noir')
#plt.imshow(gray,cmap='gray')
#
#plt.subplot(224)
#plt.axis('off')
#plt.title('Bilateral-Noir')
#plt.imshow(noir_bilateral,cmap='gray')
#
#plt.tight_layout()
#
#plt.show()

#def calculateDistance(i1, i2):
#    return np.sum((i1-i2)**2)
#
#img_gt = cv2.imread(r'C:\SAI\IIIT\2019_Monsoon\DIP\Assignment2\input_data\gt_sky.png')
#rgb_gt,gray_gt=getColorSpaces(img_gt)
#
#  
#    
#vals = [3,5,7]
#for i in range(3):
#    fig = plt.figure(figsize=(16,8))
#    for j in range(3):
#        ax = fig.add_subplot(1,3,j + 1)
#        sky_bilateral=applyBilateralFilter(gray,vals[i], vals[j])
#        ax.imshow(sky_bilateral, cmap='gray')
#        ax.axis('off')
#        ax.set_title('ad={} ar={} '.format(vals[i],vals[j]))
#plt.show()


