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

# Blur and convert to HSV
image_blur = cv2.GaussianBlur(img, (15, 15), 5)
#show(image_blur)
image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_BGR2HSV)

# Filter based on green hue
min_green = np.array([20, 40, 40])
max_green = np.array([100, 256, 256])
image_green = cv2.inRange(image_blur_hsv, min_green, max_green)

thresh = cv2.adaptiveThreshold(image_green,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

#img = img[:,:,1]

mser = cv2.MSER_create(_delta=5,
						_min_area=100,
						_max_area=14400,
						_max_variation=.25,
						_min_diversity=.2,
						_max_evolution=200,
						_area_threshold=1.01,
						_min_margin=0.003,
						_edge_blur_size=5)

vis = img.copy()
regions = mser.detectRegions(thresh, None)
#print(regions[10])
#hulls = [cv2.convexHull(regions[10].reshape(-1, 1, 2)) ] #for p in regions]
#cv2.polylines(vis, hulls, 1, (0, 255, 0))

for region in regions:
	[x,y,w,h] = cv2.boundingRect(region)
	cv2.rectangle(vis,(x,y),(x+w,y+h),(0,0,255),2)

#show(vis)
#show(thresh)
show(image_green)
