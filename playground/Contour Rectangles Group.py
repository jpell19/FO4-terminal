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

#show(image_blur)

# Adaptive threshold on green channel
thresh = cv2.adaptiveThreshold(image_blur[:,:,1], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

#img = img[:,:,1]

im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

show(im2)

vis = img.copy()

# Add some overlap to bounding boxes
for contour in contours:
	[x,y,w,h] = cv2.boundingRect(contour)
	cv2.rectangle(vis,(x,y),(x+w,y+h),(0,0,255),2)

show(vis)
#show(thresh)
#show(image_green)

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
