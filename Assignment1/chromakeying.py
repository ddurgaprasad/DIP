# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 14:41:16 2019

@author: E442282
"""

import os,sys
import cv2
from matplotlib import pyplot as plt
import numpy as np


def getColorSpaces(image):
    rgb = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    return rgb,gray

def getImageDimnesion(image):
    height,width = image.shape[:2]

    return height,width

def showImage(image,title):
    plt.imshow(image)
    plt.axis('off')
    plt.title(title)

       
def mergeImage(green_frame,target_frame):
    
         
    lower_green = np.array([50])
    upper_green = np.array([70]) 

    height, width, bands = target_frame.shape
    rows,cols,channels = green_frame.shape
    # cols-1 and rows-1 are the coordinate limits.
    
    #Translating fore- ground to match background video context
    M = np.float32([[1,0,200],[0,1,100]])
#    green_frame = cv2.warpAffine(green_frame,M,(cols,rows), borderMode=cv2.BORDER_CONSTANT,
#                               borderValue=(0,255,0))

    # BGR to HSV
    hsv = cv2.cvtColor(green_frame, cv2.COLOR_BGR2HSV)
    # HSV bands
    h = hsv[:,:,0]
    #s = hsv[:,:,1]
    #v = hsv[:,:,2]
    
    #Threshold the HSV image to get only green color
    #hue_mask_source is 255 where color between lower_green & upper_green [background]
    #otherwise it is zero [foreground]
    hue_mask_source = cv2.inRange(h, lower_green, upper_green) 
            
    #Generate Panchromatic image by inverting the values of FG and BG of hue_mask_source
    #Background is 0
    #Foreground is 255
    hue_masked_source_image=np.where(hue_mask_source != 0,0,255)
    
   
    # From original green screen image, identify the foreground identified by using hue_mask_source as cookie cutter
    masked_image = np.copy(green_frame)
    masked_image[hue_masked_source_image == 0] = [0, 0, 0]

    #Make a hole in the target image to the extent of source FG boundary
    output_video_frame = target_frame.copy()
    #output_video_frame=cv2.resize(output_video_frame,(640,360))
    output_video_frame[hue_masked_source_image != 0] = [0, 0, 0]
    #Add Source FG and background image
    output_video_frame = output_video_frame + masked_image
    
    #output_video_frame=cv2.resize(output_video_frame,(640,360))
       
    print('\nCompleted Chorma keying')
    
    return output_video_frame
     
images=[]
titles=[]
       
def main(argv):
    
   
   
   fg=r'C:/SAI/IIIT/2019_Monsoon/DIP/Assignment1/A1_resources/DIP_2019_A1/fg.jpg'
   bg=r'C:/SAI/IIIT/2019_Monsoon/DIP/Assignment1/A1_resources/DIP_2019_A1/bg.jpg'
   
   green_frame=cv2.imread(fg)
   green_frame_rgb,_=getColorSpaces(green_frame)
     
   
   target_frame=cv2.imread(bg)
   target_frame_rgb,_=getColorSpaces(target_frame)
   
   
   assert(green_frame.shape==target_frame.shape)
   
   merged_image=mergeImage(green_frame_rgb,target_frame_rgb)
   
   
   images.append(green_frame_rgb)
   images.append(target_frame_rgb)
   images.append(merged_image)
   titles.append("Foregroud")
   titles.append("Background")
   titles.append("Merged")


   num=0
   plt.figure(figsize=(8, 12))
        
   for img,title in zip(images,titles):
       plt.subplot(1,3,num+1)
       plt.axis('off')
       plt.title(title)   
       plt.imshow(img)
       num=num+1
   
   
if __name__ == "__main__":
   main(sys.argv[1:])
   




















   