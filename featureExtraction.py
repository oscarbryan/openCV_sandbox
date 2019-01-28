import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
#mport matplotlib.pyplot as plt
#import matplotlib as mpl
#mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
def ImportImage(filename):
    try:
        im= cv2.imread(filename)
    except:
        print("incompatible image file")
        exit()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im
def ObjectExtraction(im):
    print("object Extraction running")
    #CLAFFIY WITH COLOUR
    #KMEANS colour clustering
    im_array = im.reshape((im.shape[0] * im.shape[1], 3)) # convert image to list of pixel values
    print(im.shape)
    k = 5 #k number of clusters
    clt = KMeans(n_clusters = k)
    clt.fit(im_array)

    #SEGMENT IMAGE INTO COLOUR CLUSTERS
    for n in range(0,k):
        im_array[clt.labels_ == n, :] = clt.cluster_centers_[n,:]
    im_kmeans= im_array.reshape((im.shape[0], im.shape[1],3))
    return im_kmeans

    #DBSCAN SEGMENTED IMAGES INTO DISCRETE OBJECTS
    for n in range(0,k):
        db[n]= DBSCAN(eps=0.3, min_samples=10).im_array[im_array == n ]
    #CLASSIFY WITH EDGES


filename = "/home/ozzy/openCV_sandbox/images/diver/3.png"
im = ImportImage(filename)
im = ObjectExtraction(im)
plt.figure()
plt.axis("off")
plt.imshow(im)
plt.show()
