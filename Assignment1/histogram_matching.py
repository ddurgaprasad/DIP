import numpy as np
import cv2
from matplotlib import pyplot as plt


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

def histMatching(im1,im2):

    h,w = im1.shape

    hist1 = getHistogram(im1)
    cum_hist1 =getCDF(hist1)

    hist2 = getHistogram(im2)
    cum_hist2 =getCDF(hist2)
    
    pixels = np.arange(256)
    new_pixels = np.interp(cum_hist1, cum_hist2, pixels) 
    new_im = (np.reshape(new_pixels[im1.ravel()], im1.shape)).astype(np.uint8)
    
    
#    new_im = np.array(im1)
#    
#    for i in range(256):
#        diff = abs(cum_hist1[i] - cum_hist2)
#        ind = np.argmin(diff)
#        new_im[new_im == i] = ind    
        
        
    return  new_im

imge_path1='A1_resources\DIP_2019_A1\eye.png'
im1 = cv2.imread(imge_path1,0)

imge_path2='A1_resources\DIP_2019_A1\eyeref.png'
im2 = cv2.imread(imge_path2,0)


new_image=histMatching(im1,im2)
#cv2.normalize(new_image, new_image, 0, 255, cv2.NORM_MINMAX)
#plt.imshow(new_image,cmap='gray')

plt.figure(figsize=(8, 8))
plt.subplots_adjust(bottom=0.5, right=1.2, top=1.2)

plt.subplot(3,3,1)
plt.title("Original Image")
plt.axis('off')
plt.imshow(im1,cmap='gray')

plt.subplot(3,3,2)
plt.title("Histogram of image to be adjusted")
#plt.axis('off')
hist_ori = getHistogram(im1)
plt.plot(hist_ori)

plt.subplot(3,3,3)
plt.title("CDF of image to be adjusted")
#plt.axis('off')
cdf_ori = getCDF(hist_ori)
plt.plot(cdf_ori)

plt.subplot(3,3,4)
plt.title("Reference Image")
plt.axis('off')
plt.imshow(im1,cmap='gray')

plt.subplot(3,3,5)
plt.title("Histogram of reference image")
#plt.axis('off')
hist_ref = getHistogram(im2)
plt.plot(hist_ref)

plt.subplot(3,3,6)
plt.title("CDF of reference image")
#plt.axis('off')
cdf_ref = getCDF(hist_ref)
plt.plot(cdf_ref)

plt.subplot(3,3,7)
plt.title("Output Image")
plt.axis('off')
plt.imshow(im1,cmap='gray')

plt.subplot(3,3,8)
plt.title("Histogram of output image")
#plt.axis('off')
hist_new = getHistogram(im2)
plt.plot(hist_new)

plt.subplot(3,3,9)
plt.title("CDF of output image")
#plt.axis('off')
cdf_new = getCDF(hist_new)
plt.plot(cdf_new)

plt.tight_layout()
plt.show()   


