import numpy as np
import cv2
from matplotlib import pyplot as plt


#plt.figure(figsize=(12, 12))
#
#plt.subplot(3,2,1)
#plt.title("Original Image")
#plt.axis('off')
#plt.imshow(image,cmap='gray')
#
#plt.subplot(3,2,2)
#plt.xlabel('Pixel Value',fontweight='bold')
#plt.ylabel('Pixel Count', fontweight='bold')
#plt.hist(image.ravel(),bins=256,range=[0,256])


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
    

imge_path='A1_resources\DIP_2019_A1\part2.png'
image = cv2.imread(imge_path,0)

image = cv2.imread(r'C:\SAI\IIIT\2019_Monsoon\DIP\Assignment2\input_data\cubes.png',0)


new_image=histEqualization(image)


plt.figure(figsize=(8, 8))
plt.subplots_adjust(bottom=0.1, right=1.5, top=0.9)

plt.subplot(2,3,1)
plt.title("Original Image")
plt.axis('off')
plt.imshow(image,cmap='gray')

plt.subplot(2,3,2)
plt.title("PDF")
#plt.axis('off')
hist_ori = getHistogram(image)
plt.plot(hist_ori)

plt.subplot(2,3,3)
plt.title("CDF")
#plt.axis('off')
cum_hist_ori = getCDF(hist_ori)
plt.plot(cum_hist_ori)

plt.subplot(2,3,4)
plt.title("Histrogram Equalization")
plt.axis('off')
plt.imshow(new_image,cmap='gray')

plt.subplot(2,3,5)
plt.title("New PDF")
#plt.axis('off')
hist_new = getHistogram(new_image)
plt.plot(hist_ori)

plt.subplot(2,3,6)
plt.title("New CDF")
#plt.axis('off')
cum_hist_new = getCDF(hist_new)
plt.plot(cum_hist_new)

plt.show()   


