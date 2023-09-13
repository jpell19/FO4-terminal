import cv2
import numpy as np
from matplotlib import pyplot as plt


def show(image, color=True):
	# Figure size in inches
	plt.figure(figsize=(15, 15))

	# Show image, with nearest neighbour interpolation
	if color:
		plt.imshow(image, interpolation='nearest')
	else:
		plt.imshow(image, cmap='gray')
	plt.show()

image_path = './Screenshots/20170918_181106.jpg'

#image_path = './Screenshots/9-18-2017_6-03-36_PM.png'

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Blur and convert to HSV
image_blur = cv2.GaussianBlur(img, (11, 11), 2)

# show(image_blur[:,:,1], False)

# Adaptive threshold on green channel
thresh = cv2.adaptiveThreshold(image_blur[:,:,1], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

mser = cv2.MSER_create(_delta = 5,
						_min_area = 100,
						_max_area = 14400,
						_max_variation = 0.25,
						_min_diversity = .2,
						_max_evolution = 200,
						_area_threshold = 1.01,
						_min_margin = 0.003,
						_edge_blur_size = 5 )


regions = mser.detectRegions(thresh, None)


[height, width] = img.shape[:2]


img_regions = np.zeros((height, width), np.uint8)

# Add some overlap to bounding boxes
for region in regions:
	for point in region:
		img_regions[point[1], point[0]] += 1

show(img_regions, color=False)

vis = img.copy()

minLineLength = 50
maxLineGap = 100
lines = cv2.HoughLinesP(img_regions,1,np.pi/180,50,minLineLength,maxLineGap)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(vis,(x1,y1),(x2,y2),(0,0,255),5)

show(vis)

im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



for contour in contours:
	[x,y,w,h] = cv2.boundingRect(contour)
	cv2.rectangle(vis,(x,y),(x+w,y+h),(0,0,255),2)

show(vis)

#box_NMS = non_max_suppression_fast(np.array(boxes), .01)

grouped_boxes = cv2.groupRectangles(boxes, 10, 10)

# print(grouped_boxes)

# dense_box_i = np.argmax(grouped_boxes[1], axis=0)

# dense_box = grouped_boxes[0][dense_box_i]

#print(dense_box)

vis2 = img.copy()

for box in grouped_boxes[0]:
	[y,x,h, w] = box
	cv2.rectangle(vis2,(x,y),(x+w,y+h),(0,0,255),2)

show(vis2)
