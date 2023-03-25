import numpy as np
import imutils as im
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2 as cv


bgr_img = cv.imread('hw3_image.jpg')
resized_img = im.resize(bgr_img, width = 400)
hsv_img = cv.cvtColor(resized_img, cv.COLOR_BGR2HSV)

hsv_lower = (68, 95, 101)
hsv_upper = (84, 138, 134)
balled_img = cv.inRange(hsv_img, hsv_lower, hsv_upper)

# ------------------------Contour Starts here------------------------
ret, thresh = cv.threshold(balled_img, 127, 255, cv.THRESH_BINARY)

_, contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(balled_img, contours, -1, (0, 0, 0), 1)
cnt = contours[0]

(x, y), radius = cv.minEnclosingCircle(cnt)
center = (int(x), int(y))
radius = int(radius)
cv.circle(resized_img, center, radius, (0, 255, 0), 2)
cv.circle(resized_img, center, 1, (0, 0, 255), 3)

# cv.imshow("RGB_HSV_Masked Merged", np.hstack([resized_img, hsv_img, balled_img]))
cv.imshow("balled image", resized_img)
cv.waitKey(0)

cv.destroyAllWindows()
