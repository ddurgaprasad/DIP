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

def getHighBoost(image,weight_factor,kernel_size=(5, 5)):
    
    sharpened_image = applyUnsharp(image,kernel_size)
    #if A = 1, it becomes it becomes “standard standard” Laplacian sharpening
    if weight_factor==1: 
        return sharpened_image
    highboost=(weight_factor-1)*image+sharpened_image
    highboost =getNormalizedImage(highboost)
    
    return highboost


def getHighBoost2(image,A,val=1):
    
    if(val==1):
        filter1 = np.array([[0,-1,0],[-1,A+4,-1],[0,-1,0]])   
    if(val==2):   
        filter1 = np.array([[-1,-1,-1],[-1,A+8,-1],[-1,-1,-1]])   
    
    img1 = cv2.filter2D(image, -1, filter1)   
    
    return img1
def linContrastStretching(image_gray,a,b):    
    c = np.amin(image_gray)
    d = np.amax(image_gray)

    out_image =image_gray- c
    out_image *= int((b-a)/(d - c))
    out_image+=a
    
    return out_image
    
img = cv2.imread(r'C:\SAI\IIIT\2019_Monsoon\DIP\Assignment2\input_data\ravva_upma.jpg')
plt.figure(figsize=(12, 12))
rgb,gray=getColorSpaces(img)


#lowpass = cv2.GaussianBlur(rgb, (3,3), 1.0) #Lowpass
#A=4
#highboost = A*rgb -lowpass

#plt.subplot(3,2,1)
#plt.axis('off')
#plt.title('Original')
#plt.imshow(rgb)
#
#plt.subplot(3,2,2)
#plt.axis('off')
#plt.title('Highboost')
#plt.imshow(getHighBoost(rgb,1))




plt.figure(figsize=(12, 12))
plt.axis('off')
plt.title('Original and High-Boost')
plt.imshow(np.hstack((rgb, getHighBoost(rgb,1))))
plt.show()

plt.figure(figsize=(12, 12))
plt.axis('off')
plt.title('Original and Bilateral')
plt.imshow(np.hstack((rgb, cv2.bilateralFilter(rgb,9,75,75))))
plt.show()


#Highboost only enhances sharp edges but not the change in intensity


#lstA=[1,1.2,1.5,1.8,2.0]
#num=0
#for A in lstA:    
#    plt.subplot(1,5,num+1) 
#    plt.axis('off')
#    plt.title('A =' +str(A))  
#    
#    highboost=getHighBoost2(rgb,A,1)     
#    contrast_enhanced_img=linContrastStretching(highboost,0,255)
#    plt.imshow(contrast_enhanced_img)
#    num=num+1
    

#lstA =[1,1.2,1.7,2.0]
#
#num=0
#for A in lstA:    
#    plt.subplot(2,4,num+1) 
#    plt.axis('off')
#    plt.title('Weight Factor =' +str(A))  
#    highboost=getHighBoost(rgb,A)      
#    plt.imshow(highboost)
#    num=num+1
#   
#A=1.4  
##num=0
#for ks in range(3,11,2):
#    
#    plt.subplot(2,4,num+1) 
#    plt.axis('off')
#    plt.title('Kernel Size =' +str(ks)+'X'+str(ks))  
#    highboost=getHighBoost(rgb,A,(ks,ks))      
#    plt.imshow(highboost)
#    num=num+1
    

#https://www.codingame.com/playgrounds/2524/basic-image-manipulation/filtering


#Note that the low spatial frequency components (global, 
#large black background and bight areas) are suppressed while the high spatial frequency components (the texture of the fur and the whiskers) are enhanced. After a linear stretch, the image on the right is obtained.


