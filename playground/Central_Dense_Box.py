import cv2
import numpy as np
from matplotlib import pyplot as plt


def show(image):
    # Figure size in inches
    plt.figure(figsize=(15, 15))
    
    # Show image, with nearest neighbour interpolation
    plt.imshow(image, interpolation='nearest')

    plt.show()


def find_boundaries_1d(v, n):
    cumsum = np.insert(np.cumsum(v), 0, np.zeros(n))
    deltas = cumsum[n:] - 2 * cumsum[:-n]
    #min_i = np.argmax(deltas, axis=0)
    #max_i = np.argmin([i for i in -deltas if i > 0], axis=0)
    return deltas

# Need to set working directory in pycharm console preferences

image_path = './Screenshots/20170918_181106.jpg'
# image_path = './Screenshots/9-18-2017_6-03-36_PM.png'

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Blur and convert to HSV
image_blur = cv2.GaussianBlur(img, (11, 11), 2)

# show(image_blur)

# Adaptive threshold on green channel
thresh = cv2.adaptiveThreshold(image_blur[:, :, 1], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


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

# region structure:
# - list of regions
# - region consists of set of points (x,y) array

# Get image shape
img_size = img.shape

# Initialize empty arrays with shape of image
row_counter = np.zeros(img_size[0])
col_counter = np.zeros(img_size[1])

# Count interest points to determine
for region in regions:
	for point in region:
		col_counter[point[0]] += 1
		row_counter[point[1]] += 1


d = find_boundaries_1d(row_counter, 10)

plt.plot(d)
plt.show()

plt.plot(row_counter)
plt.show()

plt.plot(col_counter)
plt.show()

[x,y,w,h] = cv2.boundingRect(regions[19])
cv2.rectangle(vis,(x,y),(x+w,y+h),(0,0,255),2)

for region in regions:
	[x,y,w,h] = cv2.boundingRect(region)
	cv2.rectangle(vis,(x,y),(x+w,y+h),(0,0,255),2)
show(vis)


