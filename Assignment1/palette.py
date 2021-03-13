# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 07:49:21 2019

@author: E442282
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import KMeans

def getColorSpaces(image):
    rgb = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    return rgb,gray

def getImageDimnesion(image):
    height,width = image.shape[:2]

    return height,width


def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    # return the histogram
    return hist

def plot_colors(hist, centroids,plot_height,plot_width):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((plot_height,plot_width, 3), dtype="uint8")
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
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        cv2.rectangle(bar, (int(endX), 0), (int(endX + margin), 50),
                      (255, 255, 255), -1)
        startX = endX + margin

    # return the bar chart
    return bar

# A helper function to resize images
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

imga_path='C:/opencv-master/samples/data/home.jpg'
image = cv2.imread(imga_path)
rgb,_=getColorSpaces(image)
height,width=getImageDimnesion(rgb)
plot_width=width
plot_height=20


clusters = 5
margin = 2
borderSize = 40
offset = 2

# Let's make a copy of this image
        # to use for the color palette generation
image_copy = image_resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width=100)

# Since the K-means algorithm we're about to do,
# is very labour intensive, we will do it on a smaller image copy
# This will not affect the quality of the algorithm
pixelImage = image_copy.reshape((image_copy.shape[0] * image_copy.shape[1], 3))

# We use the sklearn K-Means algorithm to find the color histogram
# from our small size image copy
clt = KMeans(n_clusters=clusters+offset)
clt.fit(pixelImage)


# build a histogram of clusters and then create a figure
# representing the number of pixels labeled to each color
hist = centroid_histogram(clt)

# Let's plot the retrieved colors. See the plot_colors function
# for more details
bar = plot_colors(hist, clt.cluster_centers_,plot_height,plot_width)



vis = np.concatenate((rgb, bar), axis=0)

plt.imshow(vis)


