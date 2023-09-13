import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt

def show(image):
    # Figure size in inches
    plt.figure(figsize=(15, 15))
    
    # Show image, with nearest neighbour interpolation
    plt.imshow(image, interpolation='nearest')

    plt.show()

#image_path = '../Screenshots/20170918_181106.jpg'

image_path = '../Screenshots/9-18-2017_6-03-36_PM.png'
img = cv2.imread(image_path)

image_blur = cv2.GaussianBlur(img, (11, 11), 2)

thresh = cv2.adaptiveThreshold(image_blur[:,:,1],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

#img = img[:,:,1]

mser = cv2.MSER_create(_delta = 11,
						_min_area = 50000,
						_max_area = 10000000,
						_max_variation = 0.5,
						_min_diversity = .5,
						_max_evolution = 200,
						_area_threshold = 1.01,
						_min_margin = 0.003,
						_edge_blur_size = 11 )

vis = img.copy()
regions = mser.detectRegions(thresh, None)
#print(regions[10])
#hulls = [cv2.convexHull(regions[10].reshape(-1, 1, 2)) ] #for p in regions]
#cv2.polylines(vis, hulls, 1, (0, 255, 0))

for region in regions:
	[x,y,w,h] = cv2.boundingRect(region)
	cv2.rectangle(vis,(x,y),(x+w,y+h),(0,0,255),2)

show(vis)
# show(thresh)

