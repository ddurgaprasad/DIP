
import numpy as np
import cv2 
import os
import sys
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
                               
def getBinaryImage(gray,thresh=127):
    ret,thresh= cv2.threshold(gray,thresh,255,cv2.THRESH_BINARY)
    return thresh
        
def getHistogramAdjusted(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    
    lab_planes = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    
    lab_planes[0] = clahe.apply(lab_planes[0])
    
    lab = cv2.merge(lab_planes)
    
    adjusted = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    

    return adjusted    
    

# # Convert BGR to HSV

def nothing(x):
    pass



image1=cv2.imread(r'C:\SAI\IIIT\2019_Monsoon\DIP\Project\input\6.jpg')
gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)    

image2=cv2.imread(r'C:\SAI\IIIT\2019_Monsoon\DIP\Project\detect_stripes_decent\6.jpg')
gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)    




alpha = 0.5

input_alpha = 0.7
if 0 <= alpha <= 1:
    alpha = input_alpha
# [load]
src1 = gray1.copy()
src2 = gray2.copy()
# [load]
if src1 is None:
    print("Error loading src1")
    exit(-1)
elif src2 is None:
    print("Error loading src2")
    exit(-1)
# [blend_images]
beta = (1.0 - alpha)
dst = cv2.addWeighted(src1, alpha, src2, beta, 0.0)

# [blend_images]
# [display]
cv2.imshow('dst', dst)
cv2.waitKey(0)
# [display]
cv2.destroyAllWindows()




