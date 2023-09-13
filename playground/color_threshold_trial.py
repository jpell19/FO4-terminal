import cv2
import matplotlib
from matplotlib import colors
from matplotlib import pyplot as plt
import numpy as np

def show(image):
    # Figure size in inches
    plt.figure(figsize=(15, 15))
    
    # Show image, with nearest neighbour interpolation
    plt.imshow(image, interpolation='nearest')

    plt.show()
    
def show_hsv(hsv):
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    show(rgb)
    
def show_mask(mask):
    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap='gray')
    
def overlay_mask(mask, image):
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    show(img)

def show_hsv_hist(image):
    # Hue
    plt.figure(figsize=(20, 3))
    histr = cv2.calcHist([image], [0], None, [180], [0, 180])
    plt.xlim([0, 180])
    colours = [colors.hsv_to_rgb((i/180, 1, 0.9)) for i in range(0, 180)]
    plt.bar(range(0, 180), histr, color=colours, edgecolor=colours, width=1)
    plt.title('Hue')

    # Saturation
    plt.figure(figsize=(20, 3))
    histr = cv2.calcHist([image], [1], None, [256], [0, 256])
    plt.xlim([0, 256])

    colours = [colors.hsv_to_rgb((0, i/256, 1)) for i in range(0, 256)]
    plt.bar(range(0, 256), histr, color=colours, edgecolor=colours, width=1)
    plt.title('Saturation')

    # Value
    plt.figure(figsize=(20, 3))
    histr = cv2.calcHist([image], [2], None, [256], [0, 256])
    plt.xlim([0, 256])

    colours = [colors.hsv_to_rgb((0, 1, i/256)) for i in range(0, 256)]
    plt.bar(range(0, 256), histr, color=colours, edgecolor=colours, width=1)
    plt.title('Value')

    plt.show()

def find_biggest_contour(image):
    
    # Copy to prevent modification
    image = image.copy()
    img, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #print len(contours)

    # Isolate largest contour
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
 
    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
    return biggest_contour, mask


# Start manipulations

image = cv2.imread('../Screenshots/20170918_181106.jpg')

#image = cv2.imread('../Screenshots/9-18-2017_6-03-36_PM.png')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Check out HSV histogram
matplotlib.rcParams.update({'font.size': 16})   
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
#show_hsv_hist(hsv)

# Blur and convert to HSV
image_blur = cv2.GaussianBlur(image, (15, 15), 5)
#show(image_blur)
image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

# Filter based on green hue
min_green = np.array([20, 40, 40])
max_green = np.array([100, 256, 256])
image_green = cv2.inRange(image_blur_hsv, min_green, max_green)

# Make kernal for morhping
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

#image_green_eroded = cv2.morphologyEx(image_green, cv2.MORPH_ERODE, kernel)
#show_mask(image_green_eroded)

dilated = cv2.dilate(image_green,kernel,iterations = 1)

# Fill small gaps
image_green_closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
show_mask(image_green_closed)

# Remove specks
#image_green_closed_then_opened = cv2.morphologyEx(image_green_closed, cv2.MORPH_OPEN, kernel)
#show_mask(image_green_closed_then_opened)

# Find biggest contour
big_contour, green_mask = find_biggest_contour(image_green_closed)

# Apply bounding rectangle
image_with_rectangle = image.copy()
rectangle = cv2.boundingRect(big_contour)
x,y,w,h = rectangle
cv2.rectangle(image_with_rectangle,(x,y),(x+w,y+h),(255,0,9),10)
#cv2.imwrite('terminal_color_filtered.jpg', image_with_rectangle)
show(image_with_rectangle)

