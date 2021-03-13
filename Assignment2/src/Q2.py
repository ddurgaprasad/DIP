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
    kernel_size=7
    
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
sharpened image = original image – blurred image
where k is a scaling constant. Reasonable values for k vary between 0.2 and 0.7, 
with the larger values providing increasing amounts of sharpening.
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

def isequal(im1,im2):    
    if im1.shape != im2.shape:
        return False
    difference = cv2.subtract(im1, im2)   
    if cv2.countNonZero(difference) == 0:
        print("The images are completely Equal")
        return True    
    else:
        return False
    
def is_similar(image1, image2):
    return image1.shape == image2.shape and not(np.bitwise_xor(image1,image2).any())        


def getNormalizedImage(image):
    norm_image=image.copy()
    norm_image = np.maximum(norm_image, np.zeros(norm_image.shape))
    norm_image = np.minimum(norm_image, 255 * np.ones(norm_image.shape))
    norm_image = norm_image.round().astype(np.uint8)
    
    return norm_image

'''

Implement high-boost filtering on the image bell.jpg varying the window size and
the weight factor and report your observations.

 Edges = Original - LoG


Unsharp Mask
 High pass/Edges= Original - Low Pass (Gaussian,average etc)
 Sharp Image =Original+ k*Edges


High Boost = A*Original- Low Pass
             =A*Original-Original+Original- Low Pass
             =Original*(A-1)+(Original- Low Pass)
             =(A-1)*Original+High Pass
         
'''

def applyUnsharp(image, kernel_size=(5, 5)):
    """Return a sharpened version of the image, using an unsharp mask."""
    k=1.0
    sigma=1.0
    blurred = cv2.GaussianBlur(image, kernel_size, sigma) #Lowpass
    sharpened = float(k + 1) * image - float(k) * blurred #gmask
    sharpened =getNormalizedImage(sharpened)

    return sharpened


img = cv2.imread(r'C:\SAI\IIIT\2019_Monsoon\DIP\Assignment2\input_data\bell.jpg')
rgb,gray=getColorSpaces(img)

gray_minus_laplacian=gray-applyLaplacian(gray)

unsharp_mask_image=applyUnsharp(gray)

laplacian=applyLaplacian(gray)
plt.figure(figsize=(20, 20))

plt.subplot(1,4,1)
plt.axis('off')
plt.title('Original')
plt.imshow(gray,cmap='gray')

#Edges are identified
plt.subplot(1,4,2)
plt.axis('off')
plt.title('LOG sigma=1.0, 7x7 kernel,')
plt.imshow(3*laplacian,cmap='gray')

#Edges are sharper , but noise also increased
plt.subplot(1,4,3)
plt.axis('off')
plt.title('Image Minus Laplacian')
plt.imshow(gray_minus_laplacian,cmap='gray')


plt.subplot(1,4,4)
plt.axis('off')
plt.title('Unsharp Mask Image')
plt.imshow(unsharp_mask_image,cmap='gray')

hist1=getHistogram(gray_minus_laplacian)

hist2=getHistogram(unsharp_mask_image)

plt.xlabel('Intensity Value')
plt.ylabel('Pixel Frequency')
plt.plot(hist1)
plt.plot(hist2)
plt.legend(['Image minus LoG','Unsharp Mask Image'], loc='upper left')

plt.show()






































