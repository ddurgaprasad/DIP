# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 08:23:16 2019

@author: E442282
"""

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


def getColorSpaces(image):
    rgb = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)    
    return rgb,gray

images_path=r'C:\SAI\IIIT\2019_Monsoon\DIP\Project\input'
images=os.listdir(images_path)

output_path=r'C:\SAI\IIIT\2019_Monsoon\DIP\Project\output'

template = cv2.imread('template1.jpg',0)

#https://stackoverflow.com/questions/11424002/how-to-detect-simple-geometric-shapes-using-opencv

for im in images[:]:
    print(im)
    img = cv2.imread(os.path.join(images_path,im))   
    
    img_rgb,gray=getColorSpaces(img)
    
    
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    
    w, h = template.shape[::-1]
    
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.5
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    
#    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img_rgb,cmap='gray')
    plt.show()

