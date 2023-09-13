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

#image_path = './Screenshots/17-09-18-1810/20170918_181106.jpg'

#image_path = './Screenshots/17-09-18-1810/9-18-2017_6-03-36_PM.png'

#image_path = './Screenshots/17-09-18-1810/20170918_180954.jpg'

image_path = './Screenshots/17-09-25-1337/20170925_133831.jpg'

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
						_max_variation = 0.25, #
						_min_diversity = .2,
						_max_evolution = 200,
						_area_threshold = 1.01,
						_min_margin = 0.003,
						_edge_blur_size = 5 )


regions = mser.detectRegions(thresh, None)

# Get image dimensions
[height, width] = img.shape[:2]

# Find image center
image_center = (height//2, width//2)

# Initialize empty image to fill with ROIs
img_regions = np.zeros((height, width), np.uint8)

# Plot regions in blank image
for region in regions:
	for point in region:
		img_regions[point[1], point[0]] += 1


show(img_regions, color=False, title="MSER Regions")

# Make kernal for morhping
open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))

# Try eroding first to get rid of small elements (e.g. thin border near terminal screen)

dilate1 = cv2.dilate(img_regions, open_kernel, iterations=2)

erode1 = cv2.erode(dilate1, open_kernel, iterations=5)

dilate2 = cv2.dilate(erode1, open_kernel, iterations=6)

show(dilate2, color=False, title="Opened")

dilated = cv2.dilate(dilate2, close_kernel, iterations=5)

eroded = cv2.erode(dilated, close_kernel, iterations=5)

show(eroded, color=False, title="Final Preprocessed")

features = eroded.copy()
rois = eroded.copy()

# Find contours
im2, contours, hierarchy = cv2.findContours(eroded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for index, contour in enumerate(contours):
	[x, y, w, h] = cv2.boundingRect(contour)
	cv2.rectangle(rois, (x, y), (x + w, y + h), (1, 1, 1), 2)
	cv2.putText(rois,str(index), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 3, (1,1,1))

show(rois, color=False, title="Dilated ROIs")

final_view = img.copy()

# Initialize contour max score tracker
best_score = 0
i_best_terminal = -1
i_potential_terminal = -1
potential_terminals = []


for index, contour in enumerate(contours):

	print(index)
	[x,y,w,h] = cv2.boundingRect(contour)

	# Calc features of box
	area = w*h
	aspect_ratio = w/h

	# Apply restrictions to terminal shape
	if 0.7 <= aspect_ratio <= 1.4 and area >= 100:

		# Calc border info
		jitter_fract = 10
		jitter_w = w//jitter_fract
		jitter_h = h//jitter_fract
		border_area_weight = int(100000*(1+1/jitter_fract)/jitter_fract)
		# border area scales linearly with ROI area scale factor - increased penalty to 100k instead of 4

		pad_features = cv2.copyMakeBorder(features, jitter_h, jitter_h,
		                         jitter_w, jitter_w, cv2.BORDER_CONSTANT, value=0)

		# Grab surround screen to ensure its feature free
		surrounding = pad_features[y: y + h + 2*jitter_h, x: x + w + 2*jitter_w]

		# Create mask to represent
		mask = np.ones_like(surrounding, dtype=np.int32)

		# Set mask borders

		# Top row
		mask[0:jitter_h, :] = -border_area_weight
		mask[-jitter_h:, :] = -border_area_weight

		# Left and right colums (minus top/bottom rows)
		mask[:, 0:jitter_w] = -border_area_weight
		mask[:, -jitter_w:] = -border_area_weight

		#roi = cv2.copyMakeBorder(features[y: y + h, x : x + w], jitter_h, jitter_h,
		#						jitter_w, jitter_w, cv2.BORDER_CONSTANT, value=0)


		roi_weighted = surrounding * mask # element wise matmul

		# sum resulting weighted ROI (max is area of roi)
		area_sum = np.sum(roi_weighted)

		# Calc center of mass
		moments = cv2.moments(surrounding)
		rect_center = (x+w//2, y + h//2)

		dist_from_center = cv2.norm(rect_center, image_center, cv2.NORM_L2)

		score = area_sum/dist_from_center**2

		if index == 97:
			show(surrounding, color=False, title="Surrounding")
			show(roi_weighted, color=False, title="Weighted Region of Interest")
			print("Pct Mask: " + str(area_sum/area))
			print("Score: " + str(score))

		# Ensure that border area has signals in less than 10% of border area
		if area_sum/area > 0.7: #0.01*border_area:
			i_potential_terminal += 1
			potential_terminals.append([x, y, w, h])

			if score > best_score:
				best_score = score
				i_best_terminal = i_potential_terminal

[tx, ty, tw, th] = potential_terminals[i_best_terminal]
cv2.rectangle(final_view, (tx,ty), (tx+tw,ty+th), (0,0,255), 2)

show(final_view, title="Final View")
