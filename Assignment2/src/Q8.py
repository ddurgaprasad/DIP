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



def applyMeanFilter(img,kernel_size):
    s=kernel_size[0]
    kernel = np.ones(kernel_size,np.float32)/(s*s)
    dst = cv2.filter2D(img,-1,kernel)    
    return dst
      

   
img = cv2.imread(r'C:\SAI\IIIT\2019_Monsoon\DIP\Assignment2\input_data\Degraded.jpg')
plt.figure(figsize=(20, 20))
rgb,gray=getColorSpaces(img)

s=3
kernel_size=(s,s)

e1 = cv2.getTickCount()
rgb_mean = cv2.blur(rgb,(5,5))
e2 = cv2.getTickCount()
time_taken = (e2 - e1)/ cv2.getTickFrequency()
print(' Time taken for mean filter ' +str(time_taken))


rgb_mean=applyMeanFilter(rgb,kernel_size)


e1 = cv2.getTickCount()
rgb_gaussain = cv2.GaussianBlur(rgb,(5,5),0)
e2 = cv2.getTickCount()
time_taken = (e2 - e1)/ cv2.getTickFrequency()

print(' Time taken for Gaussian filter ' +str(time_taken))

e1 = cv2.getTickCount()
rgb_median = cv2.medianBlur(rgb,5)
e2 = cv2.getTickCount()
time_taken = (e2 - e1)/ cv2.getTickFrequency()
print(' Time taken for Median filter ' +str(time_taken))

e1 = cv2.getTickCount()
rgb_bilateral = cv2.bilateralFilter(rgb_mean,3,1,1)
e2 = cv2.getTickCount()
time_taken = (e2 - e1)/ cv2.getTickFrequency()
print(' Time taken for Bilateral filter ' +str(time_taken))

#plt.subplot(1,5,1) 
#plt.axis('off')
#plt.imshow(rgb)
#plt.title('Original')  

#plt.subplot(1,5,2) 
#plt.axis('off')
#plt.imshow(rgb_mean)
#plt.title('Mean ' +str(s)+'X'+str(s))  

#plt.subplot(1,5,3) 
#plt.axis('off')
#plt.imshow(rgb_gaussain)
#plt.title('Gaussian ')  
#
#plt.subplot(1,5,4) 
#plt.axis('off')
#plt.imshow(rgb_median)
#plt.title('Median' )  

#plt.subplot(1,5,5) 
#plt.axis('off')
#plt.imshow(rgb_bilateral)
#plt.title('Bilateral')  

images = np.concatenate((rgb_median, rgb_gaussain), axis=1)

plt.imshow(images)
plt.title('Median on left and Gaussian on right' )  
plt.axis('off')

plt.show()


























  