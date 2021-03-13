import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


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

    
def getBinaryImage(gray,thr=127):
    ret,thresh= cv2.threshold(gray,thr,255,cv2.THRESH_BINARY)
    return thresh

def printAngle(calculatedRect):    
    if(calculatedRect.size.width < calculatedRect.size.height):        
        print("Angle along longer side: %7.2f\n", calculatedRect.angle+180);
    else:        
        print("Angle along longer side: %7.2f\n", calculatedRect.angle+90);


def getContours(image):
    
    img=image.copy()    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   
    h,w=gray.shape
    binary=getBinaryImage(gray)  
    binary[binary>=1] = 255
    binary[binary<1] = 0
    ret, labels = cv2.connectedComponents(binary)
    
    components=imshow_components(labels)  
    print('components ',len(components))
    
    
    _,contours,h = cv2.findContours(binary,1,2)
    cv2.drawContours(img,contours,0,(0,255,255),-1)    

    for cnt in contours:
        
#        rect=cv2.minAreaRect(cnt)
#        box = cv2.boxPoints(rect)
#        box_d = np.int0(box)
        
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
#        print( len(approx))
        if len(approx)==5:
#            print( "pentagon")
            cv2.drawContours(img,[cnt],0,(0,0,255),-1)
#            cv2.drawContours(img, [box_d], 0, (0,255,0), 3)
#        elif len(approx)==3:
##            print ("triangle")
#            cv2.drawContours(img,[cnt],0,(0,0,255),-1)
        elif len(approx)==4:
#            print( "rectangle/square")
            cv2.drawContours(img,[cnt],0,(0,0,255),-1)
#        elif len(approx) == 9:
##            print( "half-circle")
#            cv2.drawContours(img,[cnt],0,(0,0,255),-1)
#        elif len(approx) > 15:
##            print( "circle")
#            cv2.drawContours(img,[cnt],0,(0,0,255),-1)
    return components ,img       


    
images_path=r'C:/SAI/IIIT/2019_Monsoon/DIP/Project/Images_Thr_And_Masks/Orientation_input'
images=os.listdir(images_path)


for im in images[:40]:
    
    print(im)
    
    image = cv2.imread(os.path.join(images_path,im))
    components,img=getContours(image)

    
    plt.axis('off')
    plt.title(im)
    plt.imshow(np.hstack((components, img))) 
    plt.show()
 

    
    