# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:58:45 2019

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

def getEdges_Default(gray,low,high):
    edges = cv2.Canny(gray,low,high)
    return edges

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
    low_contrast_mask=np.zeros((height,width,3), np.bool)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
    np.copyto(sharpened, image, where=low_contrast_mask)
    
    return sharpened      


def getMagnitude(gray):
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    
    magnitude=np.sqrt(abs_sobelx*abs_sobelx+abs_sobely*abs_sobely)
    
    return magnitude,np.arctan2(abs_sobely,abs_sobelx)

    


def applySobel(gray):
    
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
    
    return img_X+img_Y

def applyPrewitt(gray):
    
    filterX = np.array([[-1,0,1],
                        [-1,0,1],
                        [-1,0,1]])
    filterY= np.array([[1,1,1],
                       [0,0,0],
                       [-1,-1,-1]])
    
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

def aaplyCanny(gray,low):
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

def getBinaryImage(gray,thr=127):
    ret,thresh= cv2.threshold(gray,thr,255,cv2.THRESH_BINARY)
    return thresh
        

def getOrientation(gray):
      gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
      gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    
      mag, ang = cv2.cartToPolar(gx, gy,angleInDegrees=True)
      
      return ang
      
def getContours(mask,img):
    _,contours,h = cv2.findContours(mask,1,2)    
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
#        print( len(approx))
        if len(approx)==5:
#            print( "pentagon")
            cv2.drawContours(img,[cnt],0,(0,0,255),-1)
        elif len(approx)==3:
#            print ("triangle")
            cv2.drawContours(img,[cnt],0,(0,0,255),-1)
        elif len(approx)==4:
#            print( "rectangle/square")
            cv2.drawContours(img,[cnt],0,(0,0,255),-1)
        elif len(approx) == 9:
#            print( "half-circle")
            cv2.drawContours(img,[cnt],0,(0,0,255),-1)
        elif len(approx) > 15:
#            print( "circle")
            cv2.drawContours(img,[cnt],0,(0,0,255),-1)
    return img       
    

def addSobel(gray,sobel):
    
     result=gray+sobel
#     result=cv2.convertScaleAbs(result)
#     
#     cv2.normalize(result, result, 0, 255, cv2.NORM_MINMAX, dtype=-1)
#     result = result.astype(np.uint8)
    
     return result
    

images_path=r'C:\SAI\IIIT\2019_Monsoon\DIP\Project\input'
images=os.listdir(images_path)

output_path=r'C:\SAI\IIIT\2019_Monsoon\DIP\Project\output_imwrite'

sobels=[]
for im in images[:]:
    print(im)
#    plt.figure(figsize=(12, 12))
    img = cv2.imread(os.path.join(images_path,im))    

#    rgb,gray=getColorSpaces(img)    
#    noisy_gray=addGaussianNoise(gray)
    
    adjusted=getHistogramAdjusted(img)
    bilateral = cv2.bilateralFilter(adjusted, 11, sigmaSpace = 75, sigmaColor =75)    
    rgb,gray=getColorSpaces(bilateral)
    
    
#    plt.subplot(2,3,1)
#    plt.axis('off')
#    plt.title('Original')
#    plt.imshow(rgb)
    
#    plt.subplot(2,3,2)
#    plt.axis('off')
#    plt.title('bilateral')
#    plt.imshow(bilateral,cmap='gray') 
    
#    plt.subplot(2,3,3)
#    plt.axis('off')
#    plt.title('Sobel')
#    plt.imshow(applySobel(gray),cmap='gray') 
    
   
    hsv = cv2.cvtColor(bilateral, cv2.COLOR_BGR2HSV)
    lab=cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
    
#    h,s,v = cv2.split(hsv)
    l,a,b = cv2.split(lab)
    
#    median = cv2.medianBlur(255-l,5)  
    
#    sobel=applySobel(l)
    
#    res=addSobel(l,sobel)
    
#    imgs_comb = np.hstack([bilateral,lab,sobel])
    
#    plt.subplot(2,3,4)
    plt.axis('off')
    plt.title('HSV-LAB')
    plt.imshow(255-l,cmap='gray') 
    
    
    
    
#    plt.imsave(os.path.join(output_path,im), imgs_comb)
    
#    cv2.imwrite(os.path.join(output_path,im), applySobel(gray))
    
#    plt.subplot(2,3,4)
#    plt.axis('off')
#    plt.title('Laplacian')
#    plt.imshow(applyLaplacian(gray),cmap='gray')
#    
#    median = cv2.medianBlur(gray,21)    
#    plt.subplot(2,3,5)
#    plt.axis('off')
#    plt.title('Median')
#    plt.imshow(median,cmap='gray')
   
#    hsv_mask=getHSVMask(bilateral)
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
#
#    getContours(mask,img)
#    plt.subplot(2,3,6)
#    plt.axis('off')
#    plt.title('Contour')
#    plt.imshow(mask,cmap='gray') 

    
#    f = np.fft.fft2(gray)
#    fshift = np.fft.fftshift(f)
#    magnitude_spectrum = 20*np.log(np.abs(fshift))
#    
#    imgs_comb = np.hstack([gray,magnitude_spectrum])
#
#
#    plt.subplot(2,3,4)
#    plt.axis('off')
#    plt.title('magnitude_spectrum')
#    plt.imshow(imgs_comb,cmap='gray')   
    
#    kernel = np.ones((5,5),np.uint8)
#    opening = cv2.morphologyEx(median, cv2.MORPH_OPEN, kernel)
#    plt.subplot(2,3,6)
#    plt.axis('off')
#    plt.title('Opening')
#    plt.imshow(opening,cmap='gray')
#    
#    print(str(getOrientation(median)[1].mean()))
    
    plt.show()
    



