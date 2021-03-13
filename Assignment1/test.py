
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

  
def LinearTransform(im, k1, k2, a, b):

     # Update input for use for main use
    k1 = k1
    k2 = int(k2*255)
    a = int(a*255)
    b = int(b*255)
    
    # Get the dimensions of background image 
    im = np.array(im)
    
    new_im = k1*im + k2
    new_im[im > b] = 255
    new_im[im < a] = 0
    new_im[new_im > 255] = 255
    new_im[new_im < 0] = 0
    
    return new_im.astype('uint8')


def display_linear_transform(im,k1,k2,a,b):
    
#   Just for displaying purposes
    x = np.arange(0,1,0.01)
    y = k1*x + k2
    y[ x < a] = 0
    y[ x > b] = 1
    y[ y > 1] = 1
    y[ y < 0] = 0

    new_im = LinearTransform(im,k1,k2,a,b)
#     plt.imshow(new_im)
    
#   Plot the graph, image
    fig = plt.figure(figsize=(16,16))
    ax1 = fig.add_subplot(1,3,2)
    ax2 = fig.add_subplot(1,3,3)
    ax3 = fig.add_subplot(1,3,1)
    ax1.imshow(np.array(im), cmap='gray')
    ax2.imshow(new_im, cmap='gray')
    ax3.plot(x,y)

    ax1.set_title('Old Image')
    ax2.set_title('Transformed Image')
    ax3.set_title('Function')
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('scaled')
    plt.show()

def multiple_linear_transform(im,k1_list,k2_list,a_list, b_list):
    #   Just for displaying purposes
    x = np.arange(0,1,0.01)
    y_list = []
    im_list = []
    for i in range(len(k1_list)):
        try:
            new_y = k1_list[i]*x + k2_list[i]
            new_y[ x < a_list[i]] = 0
            new_y[ x > b_list[i]] = 1
            
            new_im = LinearTransform(im,k1_list[i],k2_list[i],a_list[i],b_list[i])
            
            y_list.append(new_y)
            im_list.append(new_im)
            
        except Exception as e:
            print("Error while producing linear transform:",e)
    
    im_list = np.array(im_list)
    new_im = np.sum(im_list,axis=0)
    new_im[ new_im > 255] = 255
    new_im[ new_im < 0] = 0
    
    
    y_list = np.array(y_list)
    y = np.sum(y_list,axis=0)
    y[ y > 1] = 1
    y[ y < 0] = 0
   
    
#   Plot the graph, image
    fig = plt.figure(figsize=(16,16))
    ax1 = fig.add_subplot(1,3,2)
    ax2 = fig.add_subplot(1,3,3)
    ax3 = fig.add_subplot(1,3,1)
#    ax1.imshow(np.array(im), cmap='gray')
#    ax2.imshow(np.array(new_im), cmap='gray')
    ax3.plot(x,y)
    ax1.set_title('Old Image')
    ax2.set_title('Transformed Image')
    ax3.set_title('Function')
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('scaled')
    plt.show()
    

imge_path=r'A1_resources/DIP_2019_A1/squares.jpg'
im = cv2.imread(imge_path)
#
#k1 = 2.0
#k2 = -0.5
#a = 0.25
#b = 0.75

#display_linear_transform(im,k1,k2,a,b)

from PIL import Image
# Example 4
k1 = [2.0,0.0]
k2 = [0.0,2.0]
a = [0.2,0.6]
b = [0.6,0.6]
im = Image.open(r'A1_resources/DIP_2019_A1/lena.jpg').convert('L')
multiple_linear_transform(im,k1,k2,a,b)

#k1 = [0.0,0.0,0.0,0.0]
#k2 = [0.25,0.25,0.25,0.25]
#a = [0.0,0.25,0.5,0.75]
#b = [1.0,1.0,1.0,1.0]
#im = Image.open(r'A1_resources/DIP_2019_A1/contrast1.jpg').convert('L')
#multiple_linear_transform(im,k1,k2,a,b)







