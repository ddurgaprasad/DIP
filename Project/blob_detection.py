import numpy as np
import cv2 
import os
import sys
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
                               
def getBinaryImage(gray,thr=127):
    ret,thresh= cv2.threshold(gray,thr,255,cv2.THRESH_BINARY)
    return thresh
  

def getHistogramAdjusted(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    
    lab_planes = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    
    lab_planes[0] = clahe.apply(lab_planes[0])
    
    lab = cv2.merge(lab_planes)
    
    adjusted = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    

    return adjusted   
      
def getHistogramAdjusted_gray(l):   
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))    
    adjusted = clahe.apply(l)      
    return adjusted   

def applySobel(gray):
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    magnitude  = abs_sobelx+abs_sobely   
    
    angle=cv2.phase(sobelx,sobely,angleInDegrees=True)

    return magnitude,angle.mean()

def applyLaplacian(gray):
    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(gray,(3,3),0)
    # Apply Laplacian operator in some higher datatype
#    Since our input is CV_8U we define ddepth = CV_16S to avoid overflow
#    laplacian = cv2.Laplacian(blur,cv2.CV_16S,ksize=3)
    # But this tends to localize the edge towards the brighter side.
#    laplacian1 = cv2.convertScaleAbs(laplacian)

    filter1 = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    laplacian1 = cv2.filter2D(blur, -1, filter1)
    laplacian1 = cv2.convertScaleAbs(laplacian1)

    return laplacian1

'''
Sharpen the image
Use a gaussian smoothing filter and subtract the smoothed version from the original 
image (in a weighted way so the values of a constant area remain constant).
'''
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


def getHSVMask(image):   
    hMin = 0
    sMin = 0
#    vMin = 210
    vMin = 180
    
    hMax = 179
    sMax = 255
    vMax = 255

    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower, upper)
    
    output = cv2.bitwise_and(image,image, mask= mask1)
    
    _,gray=getColorSpaces(output)
        
    ret_thresh,im_bw = cv2.threshold(gray,220,255,cv2.THRESH_BINARY)
    
    return output,im_bw

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    return labeled_img

def strokeEdges(src, dst, blurKsize = 7, edgeKsize = 5):
   if blurKsize >= 3:
       blurredSrc = cv2.medianBlur(src, blurKsize)
       graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)
   else:
       graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
   cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize = edgeKsize)
   normalizedInverseAlpha = (1.0 / 255) * (255 - graySrc)
   channels = cv2.split(src)
   for channel in channels:
       channel[:] = channel * normalizedInverseAlpha
   cv2.merge(channels, dst)
   
    
#images_path=r'C:\SAI\IIIT\2019_Monsoon\DIP\Project\input'
#images=os.listdir(images_path)

images_path=r'C:\SAI\IIIT\2019_Monsoon\DIP\Project\input'
images=os.listdir(images_path)

output_path=r'C:\SAI\IIIT\2019_Monsoon\DIP\Project\output_hsv'


for im in images[:]:
    
    img = cv2.imread(os.path.join(images_path,im))   
    img=unsharp_mask(img)
    img=getHistogramAdjusted(img)
    

    lab=cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)    
    l_adjusted=getHistogramAdjusted_gray(l)
#        
#    lab = cv2.merge([l_adjusted,a,b])    
#    img_adjusted = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)   
    output,hsv_mask=getHSVMask(img)
    
    cv2.imwrite(os.path.join(output_path,im), output)
    
#    
#    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#    lab=cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#    h,s,v = cv2.split(hsv)
#    l,a,b = cv2.split(lab)
#    
#    laplacian=applyLaplacian(l)
 
#    plt.figure(figsize=(8, 8))
#    plt.axis('off')
#    plt.title('Luminance')
#    plt.imshow(l,cmap='gray') 
#    
#    plt.axis('off')
#    plt.title(im+'  Histogram Equalized')
#    plt.imshow(l_adjusted,cmap='gray')    
#    
#    
    
#    imgs_comb = np.hstack([l,l_adjusted,dilation,closing,erosion]) 
    
#
#    plt.axis('off')
#    plt.title(im+' Unsharp Masking &  Histogram Equalized')
#    plt.imshow(img) 
#    plt.show()



#    bilateral = cv2.bilateralFilter(laplacian, 9, 75, 75) 
#    vis = l_adjusted.copy()
#    mser = cv2.MSER_create()
#    regions, boundingBoxes = mser.detectRegions(l_adjusted)
#
#    for box in boundingBoxes:
#            x, y, w, h = box;
#            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 1)
#            
    
#    _,contours,h = cv2.findContours(bilateral,1,2)
##    for cnt in contours:
##        rect = cv2.minAreaRect(cnt)
##        box = cv2.boxPoints(rect)
##        box = np.int0(box)
##        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
##        if cv2.contourArea(cnt)>1000 and  len(approx)==4 :
##            
##            cv2.drawContours(img,[box],0,(0,0,255),2)
#        
#    for cnt in contours:
#        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
##        area = cv2.contourArea(cnt)
##        if(area>=100):
#
##        print( len(approx))
#        if len(approx)==5:
##            print( "pentagon")
#            cv2.drawContours(img,[cnt],0,(0,0,255),-1)
#        elif len(approx)==3:
##            print ("triangle")
#            cv2.drawContours(img,[cnt],0,(0,0,255),-1)
#        elif len(approx)==4:
##            print( "rectangle/square")
#            cv2.drawContours(img,[cnt],0,(0,0,255),-1)
#        elif len(approx) == 9:
##            print( "half-circle")
#            cv2.drawContours(img,[cnt],0,(0,0,255),-1)
#        elif len(approx) > 15:
##            print( "circle")
#            cv2.drawContours(img,[cnt],0,(0,0,255),-1)
#  
#
#    
#    imgs_comb = np.hstack([laplacian,bilateral])

#    plt.axis('off')
#    plt.title(im)
#    plt.imshow(bilateral,cmap='gray') 
#    plt.show()
# 
#
#    mat = np.argwhere(laplacian != 0)
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
#    cv2.circle(laplacian, center, 5, 255)
#    cv2.line(laplacian, center, endpoint1, (255, 255, 0) ,2)
#    cv2.line(laplacian, center, endpoint2, (255, 255, 0),2)
#    



#    plt.axis('off')
#    plt.title(im)
#    plt.imshow(output) 
#    plt.show()
 








