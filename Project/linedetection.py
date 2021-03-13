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

def showImage(image,title,cmap):
    plt.imshow(image,cmap=cmap)
    plt.axis('off')
    plt.title(title)


def splitRGBChannels(image):
  red, green, blue= cv2.split(image)
  
  return red, green, blue
                               
def getBinaryImage(gray):
    ret,thresh= cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    return thresh
        
#img = cv2.imread(r'C:\SAI\IIIT\2019_Monsoon\DIP\Assignment2\input_data\output_49.jpg')
img = cv2.imread('C:/SAI/IIIT/2019_Monsoon/DIP/Project/input/3.jpg')



#plt.figure(figsize=(12, 12))

rgb,gray=getColorSpaces(img)


plt.axis('off')
plt.imshow(img)
plt.show()

kernel = np.ones((5,5),np.uint8)
closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
#plt.figure(figsize=(12, 12))

plt.axis('off')
plt.imshow(closing,cmap='gray')
plt.show()


bottom_hat=gray-closing


#plt.figure(figsize=(12, 12))

plt.axis('off')
plt.imshow(bottom_hat,cmap='gray')
plt.show()


blur = cv2.medianBlur(bottom_hat,5)


threshold = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)


#plt.figure(figsize=(12, 12))

plt.axis('off')
plt.imshow(threshold,cmap='gray')
plt.show()

#opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)


mask=getBinaryImage(threshold)
#kernel = np.ones((3,3),np.uint8)
#plt.figure(figsize=(12, 12))

plt.axis('off')
plt.imshow(mask,cmap='gray')
plt.show()


dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)

#plt.figure(figsize=(12, 12))

plt.axis('off')
plt.imshow(dst)
plt.show()










