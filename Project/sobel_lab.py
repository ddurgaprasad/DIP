
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
    magnitude  = abs_sobelx+abs_sobely   
    
    angle=cv2.phase(sobelx,sobely,angleInDegrees=True)

    return magnitude,angle.mean()



images_path=r'input'
images=os.listdir(images_path)



for im in images[:]:
    print(im)
#    plt.figure(figsize=(12, 12))
    img = cv2.imread(os.path.join(images_path,im))   
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab=cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    h,s,v = cv2.split(hsv)
    l,a,b = cv2.split(lab)
    

    sobel,angle= applySobel(l)

    
#    imgs_comb = np.hstack([img,lab,sobel])

 
    plt.axis('off')
    plt.title('img')
    plt.imshow(img,cmap='gray') 
    plt.show()
 
    plt.axis('off')
    plt.title('lab')
    plt.imshow(lab,cmap='gray') 
    plt.show()
     
    plt.axis('off')
    plt.title('sobel')
    plt.imshow(sobel,cmap='gray') 
    plt.show()

#    plt.axis('off')
#    plt.title('hstack')
#    plt.imshow(imgs_comb,cmap='gray') 
#    plt.show()