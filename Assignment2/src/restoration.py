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

def applyPrewitt(gray):
    
    filterX = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    filterY= np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    
    img_X = cv2.filter2D(gray, -1, filterX)
    img_Y = cv2.filter2D(gray, -1, filterY)
    
    return img_X+img_Y

def applySobel(gray):
    
    filterX = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    filterY= np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    
    img_X = cv2.filter2D(gray, -1, filterX)
    img_Y = cv2.filter2D(gray, -1, filterY)
    
    return img_X+img_Y

def applyRoberts(gray):
    
    filterX = np.array([[0,1],[-1,0]])
    filterY = np.array([[1,0],[0,-1]])
    img_X = cv2.filter2D(gray, -1, filterX)
    img_Y = cv2.filter2D(gray, -1, filterY)
    
    return img_X+img_Y

def applyLaplacian(gray):
    
    filterX = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    filterY = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    img_X = cv2.filter2D(gray, -1, filterX)
    img_Y = cv2.filter2D(gray, -1, filterY)
    
    return img_X+img_Y


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
    
    
    

img = cv2.imread(r'C:\SAI\IIIT\2019_Monsoon\DIP\Assignment2\input_data\Degraded.jpg')
plt.figure(figsize=(20, 20))
rgb,gray=getColorSpaces(img)

#plt.subplot(2,3,1)
#plt.axis('off')
#plt.title('Degraded')
#plt.imshow(rgb)

# Averaging 3x3 filter
blur =   cv2.bilateralFilter(rgb,15,80,80)
plt.subplot(121),plt.imshow(rgb),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()














