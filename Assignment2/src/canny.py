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
       

def getHistogram(image, bins=256):
    
    image_pixels=image.flatten()
    # array with size of bins, set to zeros
    histogram = np.zeros(bins)
    
    # loop through pixels and sum up counts of pixels
    for pixel in image_pixels:
        histogram[pixel] += 1
    
    # return our final result
    return histogram
def getCDF(hist):
    
    cs=np.cumsum(hist)
    # numerator & denomenator
    nj = (cs - cs.min()) * 255
    N = cs.max() - cs.min()
    
    # re-normalize the cdf
    cs = nj / N
    cum_hist = cs.astype('uint8')
    
    return cum_hist


def histEqualization(im):

    h,w=im.shape
    hist = getHistogram(im)
    cum_hist = getCDF(hist)

    img_new = cum_hist[im.flatten()]
    img_new = np.reshape(img_new, im.shape)
    
    return img_new

def local_histogram_equalization(im, k=5, verbose=True):
	new_im = np.zeros_like(im)	

	try:
		h,w,ch = im.shape
	except ValueError:
		h,w = im.shape
		ch =1

	im = np.pad(im,k//2,'constant',constant_values=0)	
	for j in range(k//2,h-k//2):
		for i in range(k//2,w-k//2):
			new_im[j,i] = histEqualization(im[j-k//2:j+k//2 , i-k//2:i+k//2])[k//2,k//2]

	if verbose:
		fig = plt.figure()
		ax11 = fig.add_subplot(2,3,1)
		ax12 = fig.add_subplot(2,3,2)
		ax13 = fig.add_subplot(2,3,3)
		ax21 = fig.add_subplot(2,3,4)
		ax22 = fig.add_subplot(2,3,5)
		ax23 = fig.add_subplot(2,3,6)

		hist = getHistogram(im)
		cum_hist = np.cumsum(hist)

		ax11.plot(np.arange(0,256,1),hist)
		ax11.set_title("Prob Distribuition")
		ax12.plot(np.arange(0,256,1),cum_hist)
		ax12.set_title("Cumalative Prob Distribuition")
		ax13.imshow(im, cmap='gray')
		ax13.set_title("After")
		ax13.axis('off')

		new_hist = getHistogram(new_im)
		new_cum_hist = np.cumsum(new_hist)

		ax21.plot(np.arange(0,256,1),new_hist)
		ax21.set_title("New Prob Distribuition")
		ax22.plot(np.arange(0,256,1),new_cum_hist)
		ax22.set_title("New CDF")
		ax23.imshow(new_im, cmap='gray')
		ax23.set_title("After")
		ax23.axis('off')

		plt.show()

	return new_im

        
img = cv2.imread(r'C:\SAI\IIIT\2019_Monsoon\DIP\Assignment2\input_data\bell.jpg')
#img = cv2.imread(r'C:\SAI\IIIT\2019_Monsoon\DIP\Assignment2\input_data\cubes.png')



plt.figure(figsize=(12, 12))

rgb,gray=getColorSpaces(img)

#local_histogram_equalization(gray)

#plt.imshow(rgb)
#plt.show()
#plt.figure(figsize=(12, 12))
#sharpened_image = unsharp_mask(rgb)
#plt.imshow(sharpened_image)
#plt.show()
#Canny recommended a upper:lower ratio between 2:1 and 3:1

#_,gray=getColorSpaces(sharpened_image)

#min_threshold = 0.66 *np.median(gray)
#max_threshold = 1.33 *np.median(gray)
#edges = cv2.Canny(gray,min_threshold,max_threshold)

#plt.imshow(edges,cmap = 'gray')
#plt.show()


ret2,th2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

blur = cv2.GaussianBlur(gray,(3,3),0)

max_threshold=ret2
min_threshold=0.5*max_threshold
edges = cv2.Canny(blur,min_threshold,max_threshold)

plt.imshow(edges,cmap = 'gray')
plt.show()












