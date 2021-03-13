# -*- coding: utf-8 -*-

import numpy as np
import cv2
from matplotlib import pyplot as plt


def getColorSpaces(image):
    rgb = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    return rgb,gray

def getImageDimnesion(image):
    height,width = image.shape[:2]

    return height,width

def getImageChannels(image):
    _,_,channels = image.shape

    return channels

def split_into_rgb_channels(image):
  '''Split the target image into its red, green and blue channels.
  image - a numpy array of shape (rows, columns, 3).
  output - three numpy arrays of shape (rows, columns) and dtype same as
           image, containing the corresponding channels. 
  '''
  red = image[:,:,2]
  green = image[:,:,1]
  blue = image[:,:,0]
  return red, green, blue



def BitQuantizeImage(img,k):
    quant_img=img.copy()
    N=2**k
    quant_img=quant_img/256
    quant_img=np.floor(quant_img* N).astype('uint8')
    return quant_img/N



imga_path='C:/SAI/IIIT/2019_Monsoon/DIP/Assignment1/A1_resources/DIP_2019_A1/quantize.jpg'
image = cv2.imread(imga_path)


cols = 7
num=0
plt.figure(figsize=(20, 20))
    
for k in range(7, 0, -1):   
    plt.subplot(cols,1,num+1)    
    plt.axis('off')
    plt.title(str(k) + " bit Quantization")    
    quant_img=BitQuantizeImage(image,k)    
    plt.imshow(quant_img)
    num=num+1



def getBitPlane(image,bit_plane):
    img_bitplane = np.mod(np.floor(image/np.power(2, bit_plane)), 2)
    return img_bitplane
    
imge_path='A1_resources/DIP_2019_A1/cameraman.png'
image = cv2.imread(imge_path,0)

num=0
for bit_plane in range(8):    
    plt.subplot(1,8,num+1) 
    plt.axis('off')
    plt.title("Bit plane " +str(bit_plane))    
    img=getBitPlane(image,bit_plane)   
    plt.imshow(img,cmap='gray')
    num=num+1
    
    

#
#imge_path='C:/SAI/IIIT/2019_Monsoon/DIP/Assignment1/A1_resources/DIP_2019_A1/gamma-corr.png'
#image = cv2.imread(imge_path)
#
#def getBitPlanes(im):
#
#    lstPlanes=[128,64,32,16,8,4,2,1]
#    fig = plt.figure()
#    for k in lstPlanes:        
#        plt.imshow(255*(im//k),cmap='gray')
#        plt.axis('off')
#        plt.show()
#        im = im - k*(im//k)
#
##getBitPlanes(image)
#def adjust_gamma(image, gamma=1.0):
#    # build a lookup table mapping the pixel values [0, 255] to
#    # their adjusted gamma values
#    invGamma = 1.0 / gamma
#    table = np.array([((i / 255.0) ** invGamma) * 255
#        for i in np.arange(0, 256)]).astype("uint8")
# 
#    # apply gamma correction using the lookup table
##    return cv2.LUT(image, table)
#    plt.figure(figsize=(4, 6))
#    plt.imshow(cv2.LUT(image, table))
#    plt.axis('off')
#    plt.show()    
#
#
#def getGammaCorrection(img,gamma):
##    quant_img = img/256
##    gamma_inv=1/gamma
##    quant_img = np.floor(quant_img**gamma_inv).astype('uint8') *255   
#    encoded = ((img / 255) ^ (1 / gamma)) * 255
#    plt.imshow(encoded)
#    plt.axis('off')
#    plt.show()
#
#lstGammas=[0.02,0.4,0.6,1,2.5,4,10,20]
#
#for gamma in lstGammas:
#    adjust_gamma(image,gammas)





