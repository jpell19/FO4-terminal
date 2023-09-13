import numpy as np
import cv2

# Load an color image in color (without alpha transparency)
img = cv2.imread('../Screenshots/20170918_181106.jpg', cv2.IMREAD_COLOR)

cv2.namedWindow('FO4 Terminal Screenshot', flags=cv2.WINDOW_NORMAL)

img_scaled = cv2.resize(img, None, fx=1/10, fy=1/10)

# Show with window name
cv2.imshow('FO4 Terminal Screenshot',img_scaled)
# Wait for user to hit key (0 specifies indefinitely)
cv2.waitKey(0)

# Destroy all windows after keystroke
cv2.destroyAllWindows()