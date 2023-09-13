import cv2
import numpy as np
from matplotlib import pyplot as plt


def show(image):
	# Figure size in inches
	plt.figure(figsize=(15, 15))

	# Show image, with nearest neighbour interpolation
	plt.imshow(image, interpolation='nearest')

	plt.show()


height = 1000
width = 1000

# rect [x, y, w, h]
rects = [[0, 0, 100, 100],
		 [50, 0, 100, 100],
		 [0, 50, 100, 100],
		 [50, 50, 100, 100],
         ]

image = np.zeros((height,width,3), np.uint8)

image2 = image.copy()

for rect in rects:
	[x, y, w, h] = rect
	cv2.rectangle(image2,(x,y),(x+w,y+h),(0,0,255),2)

show(image2)

grouped_boxes = cv2.groupRectangles(rects, 2, .5)

print(grouped_boxes)

image3 = image.copy()

for box in grouped_boxes[0]:
	[x, y, w, h] = box
	cv2.rectangle(image2,(x,y),(x+w,y+h),(0,0,255),2)

show(image2)