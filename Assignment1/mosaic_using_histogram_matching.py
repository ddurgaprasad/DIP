import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

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

reference_image_path='A1_resources\DIP_2019_A1\part1.png'
reference_image = cv2.imread(reference_image_path,0)

images_path=r'A1_resources\DIP_2019_A1'
images=['part2.png','part3.png','part4.png']
images=[os.path.join(images_path,filename) for filename in images]

matched_images=[]
for imge_path in images:
    image = cv2.imread(imge_path,0)
    print('Matching ',imge_path)
    new_image=histMatching(image,reference_image)
#    new_image=histEqualization(image)
    matched_images.append(new_image)
    
top=cv2.hconcat([reference_image,matched_images[0]])
bottom=cv2.hconcat([matched_images[1],matched_images[1]])

h,w =top.shape

bottom = cv2.resize(bottom, (w,h), interpolation = cv2.INTER_AREA)
fig = plt.figure(figsize=(16,16))

mosaic=cv2.vconcat([top,bottom])
plt.title("Retrieve Original Image")
plt.imshow(mosaic,cmap='gray')
plt.axis('off')
plt.show()   

def linContrastStretching(image_gray,a,b):    
    c = np.amin(image_gray)
    d = np.amax(image_gray)

    out_image =image_gray- c
    out_image *= int((b-a)/(d - c))
    out_image+=a
    
    return out_image




fig = plt.figure(figsize=(16,16))
tl=histEqualization(reference_image)
tr=histEqualization(cv2.imread(images[0],0))
bl=histEqualization(cv2.imread(images[1],0))
br=histEqualization(cv2.imread(images[2],0))

top=cv2.hconcat([tl,tr])
bottom=cv2.hconcat([bl,br])

h,w =top.shape

bottom = cv2.resize(bottom, (w,h), interpolation = cv2.INTER_AREA)

mosaic=cv2.vconcat([top,bottom])
plt.title("Retrieve Original Image")
plt.imshow(mosaic,cmap='gray')
plt.axis('off')
plt.show() 







