import numpy as np
import cv2
from matplotlib import pyplot as plt

def getBitPlane(image,bit_plane):
    img_bitplane = np.mod(np.floor(image/np.power(2, bit_plane)), 2)
    return img_bitplane.astype('uint8')


imge_path='A1_resources/DIP_2019_A1/cameraman.png'
image = cv2.imread(imge_path,0)

plt.title("Original Image")
plt.axis('off')
plt.imshow(image,cmap='gray')
plt.show()

plt.figure(figsize=(20, 20))
num=0
images=[]

for bit_plane in range(8):    
    plt.subplot(1,8,num+1) 
    plt.axis('off')
    plt.title("Bit plane " +str(bit_plane))    
    img=getBitPlane(image,bit_plane)   
    images.append(img)
    plt.imshow(img,cmap='gray')
    num=num+1  
    
h,w=image.shape

msb_img = np.zeros((h, w), np.uint8)
lsb_img = np.zeros((h, w), np.uint8)
    


new_img = (2 * (2 * (2 * (2 * (2 * (2 * (2 * images[7] + images[6]) 
+ images[5]) + images[4]) + images[3]) + images[2]) + images[1]) + images[0]); 

lsb_img=  (2 * (2 * (2 * (2 * (2 * (2 * images[7] + images[6]) 
+ images[5]) + images[4]) + images[3]) + images[2]) + images[1]) 

msb_img = (2 * (2 * (2 * (2 * (2 * (2 * (images[6]) 
+ images[5]) + images[4]) + images[3]) + images[2]) + images[1]) + images[0]); 


plt.figure(figsize=(8, 8))

plt.subplot(3,2,1)
plt.title("Original Image")
plt.axis('off')
plt.imshow(msb_img,cmap='gray')

plt.subplot(3,2,2)
plt.xlabel('Pixel Value',fontweight='bold')
plt.ylabel('Pixel Count', fontweight='bold')
plt.hist(image.ravel(),256,[0,256])

plt.subplot(3,2,3)
plt.title("MSB set to zero")
plt.axis('off')
plt.imshow(msb_img,cmap='gray')

plt.subplot(3,2,4)
plt.xlabel('Pixel Value',fontweight='bold')
plt.ylabel('Pixel Count', fontweight='bold')
plt.hist(msb_img.ravel(),256,[0,256])


plt.subplot(3,2,5)
plt.title("LSB set to zero")
plt.axis('off')
plt.imshow(lsb_img,cmap='gray')

plt.subplot(3,2,6)
plt.xlabel('Pixel Value',fontweight='bold')
plt.ylabel('Pixel Count', fontweight='bold')
plt.hist(lsb_img.ravel(),256,[0,256])

plt.show()    













