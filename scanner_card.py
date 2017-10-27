from transform import four_point_transform
import imutils
from skimage.filters import threshold_adaptive
import numpy as np
import cv2

def scan(image):
	# PHASE 1
	ratio = image.shape[0] / 500.0
	orig = image.copy()
	image = imutils.resize(image, height = 500)
	
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 75, 200)

	# PHASE 2
	(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
	
	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		print len(approx)
		if len(approx) == 4:
			screenCnt = approx
			break

	# # PHASE 3
	warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
	warped = threshold_adaptive(warped, 251, offset = 10)
	warped = warped.astype("uint8") * 255
	
	return warped