import numpy as np
import cv2
from matplotlib import pyplot as plt
from operator import itemgetter


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

def getDominantChannel(img):
    
    r,g,b=split_into_rgb_channels(img)
    red_values_sum=np.sum(r.flatten())
    green_values_sum=np.sum(g.flatten())
    blue_values_sum=np.sum(b.flatten())
    
    dic={}
    dic['RED']=red_values_sum
    dic['GREEN']=green_values_sum
    dic['BLUE']=blue_values_sum

    return sorted(dic.items(), key=itemgetter(1), reverse=True)[0][0]


#if __name__ == "__main__":
#    
#    #path = sys.argv[1]
#    
#    img = cv2.imread( r'C:\SAI\IIIT\2019_Monsoon\DIP\Assignment1\A1_resources\DIP_2019_A1\fg.jpg')
#    
#    img = cv2.imread(r'C:/opencv-master/samples/data/home.jpg')
# 
#    
#    plt.axis('off')
#    plt.title('Dominant Channel is ' + getDominantChannel(img))
#    rgb,_=getColorSpaces(img)
#    plt.imshow(rgb)
#    plt.show()

img = cv2.imread( r'C:\SAI\IIIT\2019_Monsoon\DIP\Assignment1\A1_resources\DIP_2019_A1\fg.jpg')
#r,g,b=split_into_rgb_channels(img)
#
#    

clusters = 5
margin = 2
borderSize = 40
offset = 2

def centroid_histogram(label):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(label)) + 1)
    (hist, _) = np.histogram(label, bins = numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist

def plot_colors(hist, centroids,plot_height,plot_width):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((plot_height, plot_width, 3), dtype = "uint8")
    startX = 0
    centroids = sorted(centroids, key=lambda x: sum(x))
    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1)
        startX = endX
    
    # return the bar chart
    return bar

def plot_colors2(hist, centroids,plot_height,plot_width,clusters):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((plot_height, plot_width, 3), dtype="uint8")
    startX = 0

    # Sort the centroids to form a gradient color look
    centroids = sorted(centroids, key=lambda x: sum(x))

    # loop over the percentage of each cluster and the color of
    # each cluster

    for (percent, color) in zip(hist, centroids[offset:]):
        # plot the relative percentage of each cluster
        # endX = startX + (percent * 300)

        # Instead of plotting the relative percentage,
        # we will make a n=clusters number of color rectangles
        # we will also seperate them by a margin
        new_length = 300 - margin * (clusters - 1)
        endX = startX + new_length/clusters
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),color.astype("uint8").tolist(), -1)
        cv2.rectangle(bar, (int(endX), 0), (int(endX + margin), 50),(255, 255, 255), -1)
        startX = endX + margin

    # return the bar chart
    return bar

imga_path='C:/opencv-master/samples/data/home.jpg'
img = cv2.imread(imga_path)

 #reshape the image to an array of Mx3 size (M is number of pixels in image)
rgb,_=getColorSpaces(img)

height,width=getImageDimnesion(img)

array_pixel = np.reshape(rgb, (-1,3))
print(array_pixel.shape)
array_pixel = np.float32(array_pixel)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

lst_num_clusters=[4,6,8,10]
#lst_num_clusters=[4]
images_result=[]
images_cluster=[]
palette_bars=[]

plot_width=width
plot_height=20

for K in lst_num_clusters:
    
    ret,labels,centers = cv2.kmeans(array_pixel,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
#    plt.axis('off')
#    plt.title('Frequnetly occuring color is: rgb({})'.format(centers[0].astype(np.int32))
#              +'No of clusters({})'.format(K))

    # Now convert back into uint8, and make original image
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    res2 = res.reshape((rgb.shape))
    images_result.append(res2)
    images_cluster.append('No Of Clusters:'+str(K))
    hist=centroid_histogram(labels)

    bar=plot_colors2(hist, centers,plot_height,plot_width,K)
    palette_bars.append(bar)
    
    
#plt.imshow(res2)
cols = len(lst_num_clusters)
num=0
plt.figure(figsize=(20, 20))
    
for img,title,palette in zip(images_result,images_cluster,palette_bars):   
    plt.subplot(len(lst_num_clusters),1,num+1)    
    plt.axis('off')
    plt.title(title)
    vis = np.concatenate((img, palette), axis=0)
    plt.imshow(vis)

    num=num+1

