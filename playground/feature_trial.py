import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load an color image in color (without alpha transparency)
img = cv2.imread('../Screenshots/20170918_181106.jpg', cv2.IMREAD_COLOR)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(gray,(x,y),3,255,-1)

plt.imshow(gray),plt.show()