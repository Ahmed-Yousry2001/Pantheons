import numpy as np
import cv2
import sys

def wb(channel, perc = 0.05):
    mi, ma = (np.percentile(channel, perc), np.percentile(channel,100.0-perc))
    channel = np.uint8(np.clip((channel-mi)*255.0/(ma-mi), 0, 255))
    return channel

image = cv2.imread("coral.jpeg", 1) # load color
imWB  = np.dstack([wb(channel, 0.05) for channel in cv2.split(image)] )

hsv = cv2.cvtColor(imWB, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (20, 20, 80), (120, 255, 255))
inv_mask = cv2.bitwise_not(mask)

coral = cv2.bitwise_and(imWB, imWB, mask=inv_mask)

cv2.imshow("image1",imWB )
cv2.waitKey(0)
cv2.destroyAllWindows()

import numpy as py
import cv2

#img = cv2.imread("image1")

#hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#mask = cv2.inRange(hsv, (30, 50, 70), (120, 255, 255))
#inv_mask = cv2.bitwise_not(mask)

#coral = cv2.bitwise_and(img, img, mask=inv_mask)



cv2.imshow("img", imWB)
cv2.imshow("mask", mask)
cv2.imshow("final", coral)
cv2.imshow("inv mask", inv_mask)
cv2.waitKey()