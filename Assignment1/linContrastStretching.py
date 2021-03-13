import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec

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

'''
   (a,b) min,max of outputput image
   (c,d) min,mac of input image
    out_pixel=(input_pixel-c)(b-a/d-c)+a
'''

def linContrastStretching(image_gray,a,b):    
    c = np.amin(image_gray)
    d = np.amax(image_gray)

    out_image =image_gray- c
    out_image *= int((b-a)/(d - c))
    out_image+=a
    
    return out_image
    
def getHistogramAdjusted(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)    
    lab_planes = cv2.split(lab)    
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))    
    lab_planes[0] = clahe.apply(lab_planes[0])    
    lab = cv2.merge(lab_planes)    
    adjusted = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)    
    return adjusted  
      

img_path=r'C:\SAI\IIIT\2019_Monsoon\DIP\Project\input\10.jpg'

#img_path=  r'C:/SAI/IIIT/2019_Monsoon/DIP/images/contrast2.jpg'
img = cv2.imread(img_path)

rgb,gray=getColorSpaces(img)
a,b=0,255

#contrast_enhanced_img=linContrastStretching(gray,a,b)
contrast_enhanced_img=getHistogramAdjusted(img)


# Display Original image with colorbar and Histogram
fig = plt.figure(figsize=(8, 9))

#gs = gridspec.GridSpec(2, 1,
#         wspace=0.0, hspace=0.0, top=0.95, bottom=0.05, left=0.17, right=0.845) 

axImage1 = fig.add_subplot(211)
axImage1.set_title('Original')
axImage1.axis('off')
im1 = axImage1.imshow(img, interpolation='None',cmap='gray')

divider = make_axes_locatable(axImage1)
axColorbar1 = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=axColorbar1, orientation='vertical')

axHist1 = divider.append_axes("right", '100%', pad='20%')
axHist1.yaxis.set_label_position("right")
axHist1.yaxis.tick_right()
axHist1.set_xlabel('Pixel Value', fontweight='bold')
axHist1.set_ylabel('Pixel Count', fontweight='bold')
axHist1.hist(img.ravel(),256,[0,256])
xmax=img.max()
axHist1.annotate('Max '+str(xmax), xy=(xmax, 0), xytext=(250, 600),rotation=45,
                   va="center", ha="right", textcoords='data',xycoords='data',                 
                  arrowprops=dict(arrowstyle="->",
                                  connectionstyle="arc3,rad=-0.2",
                                  fc="w"),  )
xmin=img.min()
axHist1.annotate('Min '+str(xmin), xy=(xmin, 0), xytext=(20,60),rotation=45,
                   va="center", ha="right", textcoords='data',xycoords='data',                 
                  arrowprops=dict(arrowstyle="->",
                                  connectionstyle="arc3,rad=0.2",
                                  fc="w"),  )

plt.subplots_adjust(wspace=0, hspace=0)

#Display image after contrast stretch
axImage2 = fig.add_subplot(212)
axImage2.set_title('After Contrast Stretch')
axImage2.axis('off')
im2 = axImage2.imshow(contrast_enhanced_img, interpolation='None',cmap='gray')

divider = make_axes_locatable(axImage2)
axColorbar2 = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=axColorbar2, orientation='vertical')

axHist2 = divider.append_axes("right", '100%', pad='20%')
axHist2.yaxis.set_label_position("right")
axHist2.yaxis.tick_right()
axHist2.set_xlabel('Pixel Value',fontweight='bold')
axHist2.set_ylabel('Pixel Count', fontweight='bold')
axHist2.hist(contrast_enhanced_img.ravel(),256,[0,256])
xmax=contrast_enhanced_img.max()
axHist2.annotate('Max'+str(xmax), xy=(xmax, 0), xytext=(250,600),rotation=45,
                   va="center", ha="right", textcoords='data',xycoords='data',                 
                  arrowprops=dict(arrowstyle="->",
                                  connectionstyle="arc3,rad=-0.2",
                                  fc="w"),  )

xmin=contrast_enhanced_img.min()
axHist2.annotate('Min'+str(xmin), xy=(xmin, 0), xytext=(20,600),rotation=45,
                   va="center", ha="right", textcoords='data',xycoords='data',                 
                  arrowprops=dict(arrowstyle="->",
                                  connectionstyle="arc3,rad=0.2",
                                  fc="w"),  )


