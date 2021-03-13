
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

def getColorSpaces(image):
    rgb = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    return rgb,gray

def getImageDimnesion(image):
    height,width = image.shape[:2]

    return height,width

def getImageChannels(image):
    _,_,channels = image.shape

    return channels

def split_into_rgb_channels(image):
  '''Split the target image into its red, green and blue channels.
  image - a numpy array of shape (rows, columns, 3).
  output - three numpy arrays of shape (rows, columns) and dtype same as
           image, containing the corresponding channels. 
  '''
  red = image[:,:,2]
  green = image[:,:,1]
  blue = image[:,:,0]
  return red, green, blue

# Read the image in greyscale
images_path=r'A1_resources\DIP_2019_A1'
images=['lena.jpg','lena1.jpg','lena2.jpg','lena3.jpg']


images=[os.path.join(images_path,filename) for filename in images]

cols = len(images)
num=0

plt.figure(figsize=(20, 20))
#for img_path in images:
#    img = cv2.imread(img_path)
#    rgb,gray=getColorSpaces(img)
#    plt.subplot(1,cols,num+1) 
#    plt.axis('off')
#    plt.title("")    
#    plt.imshow(img)
#    num=num+1 

def getImageNegative(img,k):
    quant_img=img.copy()
    N=2**k
    quant_img=quant_img/256
    quant_img=np.floor(quant_img* N).astype('uint8')
    negative_img=quant_img/N
    return N-negative_img

def performGammaCorrection(img,gamma):

    gamma_corrected = ((img/255) **gamma)    
#    gamma_corrected = cv2.pow(img/255,gamma)
    return gamma_corrected

imge_path=r'A1_resources/DIP_2019_A1/gamma-corr.png'
image = cv2.imread(imge_path)
cols = 7
num=0
plt.figure(figsize=(20, 20))
num=0

lst_gammas=[0.04,0.10,0.20,0.40,0.67,1,1.5,2.5,5.0,10.0,25.0]
    
for gamma in lst_gammas:   
    plt.subplot(1,len(lst_gammas),num+1)    
    plt.axis('off')
    plt.title(str(gamma) + " gamma")    
    gamma_corrected=performGammaCorrection(image,gamma)    
    plt.imshow(gamma_corrected)
    num=num+1
plt.show()

xoffest=-100
for gamma in lst_gammas:   
    gamma_corrected=performGammaCorrection(image,gamma)    
    plt.xlabel('Input Gray level-r')   
    plt.ylabel('Output Gray level-s')
#    plt.axis('off')    
    plt.annotate(r'$\gamma$='+str(gamma), xy=(image.ravel().mean()+xoffest,gamma_corrected.ravel().mean()), xycoords='data')
    
    plt.plot(np.sort(image.ravel()),np.sort(gamma_corrected.ravel()))
    xoffest+=10
plt.show()








