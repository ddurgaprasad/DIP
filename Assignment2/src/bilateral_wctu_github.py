import numpy as np
import cv2
import time
from matplotlib import pyplot as plt




def getImageDimnesion(image):
    height,width = image.shape[:2]
    
    return height,width


def getPaddedImage(image,pad,channels,data_type):
   
    if channels==3:    
        return  np.pad(image, ((pad, pad), (pad, pad), (0, 0)), 'symmetric').astype(data_type)
       
    else:
        return np.pad(image, ((pad, pad), (pad, pad)), 'symmetric').astype(data_type)
    
def getResizedImage(img,scale_percent = 50 ):
    
    # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    return resized    

def getGaussianKernel(kernel_size=3,sigma=1.0):
    
    pad=kernel_size//2    
    ax=np.linspace(-pad,pad,kernel_size)
    ay=np.linspace(-pad,pad,kernel_size)    
    np.arange(2 * pad + 1) - pad
    X,Y = np.meshgrid(ax,ay)     
    kernel = np.exp(-(np.square(X)+np.square(Y))/(2*np.square(sigma)))

    return kernel

def applyCrossBilateralFilter(image1, image2, sigma_domain, sigma_range):
   pad = int(np.ceil(3 * sigma_domain))
   if image1.ndim == 3:  
       channels=3    
   if image1.ndim == 2:
       channels=2        
   h,w=getImageDimnesion(image1)
   image1_padded = getPaddedImage(image1,pad,channels,np.float32)
   image2_padded = getPaddedImage(image2,pad,channels,np.int32)
        
   output = np.zeros_like(image1)
  
    # A lookup table for range kernel
   
   LUT = np.exp(-np.arange(256) * np.arange(256) * 1 / (2 * sigma_range**2))
    # Generate a spatial Gaussian function
   x, y = np.meshgrid(np.arange(2 * pad + 1) - pad, np.arange(2 * pad + 1) - pad)
   
   kernel_domain = np.exp(-(x * x + y * y) * 1 / (2 * sigma_domain**2))
   
   if image1_padded.ndim == 2:
        for y in range(pad, pad + h):
            for x in range(pad, pad + w):
                W = LUT[np.abs(image2_padded[y - pad:y + pad + 1, x - pad:x + pad + 1] - image2_padded[y, x])] * kernel_domain
                output[y - pad, x - pad] = np.sum(W * image1_padded[y - pad:y + pad + 1, x - pad:x + pad + 1]) / np.sum(W)
                
   if image1_padded.ndim == 3 :  
        for y in range(pad, pad + h):
            for x in range(pad, pad + w):
                W = LUT[abs(image2_padded[y - pad:y + pad + 1, x - pad:x + pad + 1, 0] - image2_padded[y, x, 0])] * \
                      LUT[abs(image2_padded[y - pad:y + pad + 1, x - pad:x + pad + 1, 1] - image2_padded[y, x, 1])] * \
                      LUT[abs(image2_padded[y - pad:y + pad + 1, x - pad:x + pad + 1, 2] - image2_padded[y, x, 2])] * \
                      kernel_domain
                Waccum = np.sum(W)
                output[y - pad, x - pad, 0] = np.sum(W * image1_padded[y - pad:y + pad + 1, x - pad:x + pad + 1, 0]) / Waccum
                output[y - pad, x - pad, 1] = np.sum(W * image1_padded[y - pad:y + pad + 1, x - pad:x + pad + 1, 1]) / Waccum
                output[y - pad, x - pad, 2] = np.sum(W * image1_padded[y - pad:y + pad + 1, x - pad:x + pad + 1, 2]) / Waccum
   return output

def getColorSpaces(image):
    rgb = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    
    return rgb,gray



plt.figure(figsize=(12, 12))
sigma_s = 3
sigma_r =10
img1 = cv2.imread(r'C:\SAI\IIIT\2019_Monsoon\DIP\Assignment2\input_data\pots_flash.jpg')
img1=getResizedImage(img1,40)
rgb1,gray1=getColorSpaces(img1)

img2 = cv2.imread(r'C:\SAI\IIIT\2019_Monsoon\DIP\Assignment2\input_data\pots_no_flash.jpg')
img2=getResizedImage(img2,40)
  
rgb2,gray2=getColorSpaces(img2)
  
tic = time.time()


#img_bf = applyCrossBilateralFilter(gray2, gray1, sigma_s, sigma_r)
#img_bf = applyCrossBilateralFilter(img2, img1, sigma_s, sigma_r)
img_bf = applyCrossBilateralFilter(img1, img1, sigma_s, sigma_r)


toc = time.time()
print('Elapsed time: %f sec.' % (toc - tic))


rgb = cv2.cvtColor(img_bf,cv2.COLOR_BGR2RGB)

plt.axis('off')
#plt.imshow(images,cmap='gray')

images = np.concatenate((rgb1, rgb2,rgb), axis=1)
#cv2.imwrite(r'C:\SAI\IIIT\2019_Monsoon\DIP\Assignment2\output_data\output1.png', images)

plt.imshow(images)


