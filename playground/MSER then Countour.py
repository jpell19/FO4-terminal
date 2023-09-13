import cv2
import numpy as np
from matplotlib import pyplot as plt


def show(image, color=True, title=""):
	# Figure size in inches
	plt.figure(figsize=(15, 15))

	# Show image, with nearest neighbour interpolation
	if color:
		plt.imshow(image, interpolation='nearest')
	else:
		plt.imshow(image, cmap='gray')

	plt.title(title)

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


show(img_regions, False)

show(1-img_regions, False)

# Make kernal for morhping
dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

dilated = cv2.dilate(img_regions, dilation_kernel, iterations=5)

eroded = cv2.erode(dilated, dilation_kernel, iterations=5)

# show(eroded, color=False)

features = eroded.copy()
rois = eroded.copy()

# Find contours
im2, contours, hierarchy = cv2.findContours(eroded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

potential_terminals = []

for index, contour in enumerate(contours):
	[x, y, w, h] = cv2.boundingRect(contour)
	cv2.rectangle(rois, (x, y), (x + w, y + h), (1, 1, 1), 2)
	cv2.putText(rois,str(index), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 3, (1,1,1))

show(rois, color=False)

final_view = img.copy()

for index, contour in enumerate(contours):

	print(index)
	[x,y,w,h] = cv2.boundingRect(contour)

	# Calc features of box
	area = cv2.contourArea(contour)
	aspect_ratio = w/h

	# Apply restrictions to terminal shape
	if 0.7 <= aspect_ratio <= 1.4 and area >= 100:

		# Calc border info
		jitter_fract = 10
		jitter_w = w//jitter_fract
		jitter_h = h//jitter_fract
		border_area = 4*w*h*(1+1/jitter_fract)/jitter_fract

		pad_features = cv2.copyMakeBorder(features, jitter_h, jitter_h,
		                         jitter_w, jitter_w, cv2.BORDER_CONSTANT, value=0)

		# Grab surround screen to ensure its feature free
		surrounding = pad_features[y: y + h + 2*jitter_h, x: x + w + 2*jitter_w]


		roi = cv2.copyMakeBorder(features[y: y + h, x : x + w], jitter_h, jitter_h,
								jitter_w, jitter_w, cv2.BORDER_CONSTANT, value=0)


		delta = surrounding - roi

		if index == 69:
			show(surrounding, color=False, title="Surrounding")
			show(roi, color=False, title="ROI")
			show(delta, color=False, title="Delta")

		count = cv2.countNonZero(delta)

		# Ensure that border area has signals in less than 10% of border area
		if count < 1: #0.01*border_area:
			potential_terminals.append([x,y,w,h])
			cv2.rectangle(final_view, (x,y), (x+w,y+h), (0,0,255), 2)

show(final_view, title="Final View")
