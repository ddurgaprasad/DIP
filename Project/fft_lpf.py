import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from math import atan2, cos, sin, sqrt, pi




def applySobel(gray):
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    magnitude  = abs_sobelx+abs_sobely   
    
    angle=cv2.phase(sobelx,sobely,angleInDegrees=True)

    return magnitude,angle.mean()

def getFFT2D(gray):
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(1+np.abs(fshift))
    
    return magnitude_spectrum


def getFFT2D_LP(img):
    
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)  # center
    
    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))    
    
    #Create HPF Mask
    # create a mask first, center square is 0, remaining all zeros
    mask = np.zeros((rows,cols,2),np.uint8)
    mask[0:rows, ccol-10:ccol+10] = 1
    
    # apply mask and inverse DFT
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    
    return img_back,magnitude_spectrum

images_path=r'C:\SAI\IIIT\2019_Monsoon\DIP\Project\input'
images=os.listdir(images_path)


def getImageDimnesion(image):
    height,width = image.shape[:2]    
    return height,width

    
def getBinaryImage(gray,thr=127):
    ret,thresh= cv2.threshold(gray,thr,255,cv2.THRESH_BINARY)
    return thresh
    
 
for im in images[:1]:
    print(im)
#    plt.figure(figsize=(12, 12))
    im=r'C:/SAI/IIIT/2019_Monsoon/DIP/Project/input/11.jpg'
    img = cv2.imread(os.path.join(images_path,im))        
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#    lab=cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#    h,s,v = cv2.split(hsv)
#    l,a,b = cv2.split(lab)
#    
#    magnitude_spectrum=getFFT2D(l)    
#
#    cv2.normalize(magnitude_spectrum, magnitude_spectrum, 0, 255, cv2.NORM_MINMAX)
#    
#    binary=getBinaryImage(magnitude_spectrum)
    
    img_back=getFFT2D_LP(gray)
    
#    imgs_comb = np.hstack([l,magnitude_spectrum])    
    plt.axis('off')
    plt.title('L & FFT')
    plt.imshow(img_back,cmap='gray') 
    plt.show()

#    ddepth =cv2.CV_64F
#    kernel_size = 5
#    blur = cv2.GaussianBlur(h, (5, 5), 0)    
#    dst = cv2.Laplacian(blur, ddepth, ksize=kernel_size)
#    abs_dst = cv2.convertScaleAbs(dst)
#
#
#    
#    plt.axis('off')
#    plt.title('FFT->Sobel')
#    plt.imshow(abs_dst,cmap='gray') 
#    plt.show()

    #From a matrix of pixels to a matrix of coordinates of non-black points.
    #(note: mind the col/row order, pixels are accessed as [row, col]
    #but when we draw, it's (x, y), so have to swap here or there)
#    mat = np.argwhere(binary != 0)
#    mat[:, [0, 1]] = mat[:, [1, 0]]
#    mat = np.array(mat).astype(np.float32) #have to convert type for PCA
#    
#    #mean (e. g. the geometrical center) 
#    #and eigenvectors (e. g. directions of principal components)
#    m, e = cv2.PCACompute(mat, mean = np.array([]))
#    
#    #now to draw: let's scale our primary axis by 100, 
#    #and the secondary by 50
#    center = tuple(m[0])
#    endpoint1 = tuple(m[0] + e[0]*100)
#    endpoint2 = tuple(m[0] + e[1]*50)
#    
#    cv2.circle(img, center, 5, 255)
##    cv2.line(img, center, endpoint1, (255, 255, 0) ,2)
#    cv2.line(img, center, endpoint2, (255, 255, 0),2)
#    
#    plt.axis('off')
#    plt.title('PCA')
#    plt.imshow(img  ,cmap='gray') 
#    plt.show()
#    
    
    
    
    
    
    
    
    
    
    
    
    