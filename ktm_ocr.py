# import the necessary packages
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import pytesseract
from PIL import Image, ImageFilter
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# initialize a rectangular (wider than it is tall) and square
# structuring kernel
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# read image and resize
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500, height=100)

# test original output
# cv2.imshow("Original output", image)
# cv2.waitKey(0)

# change color to gray
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# test gray output
# cv2.imshow("Gray output", gray)
# cv2.waitKey(0)

# apply a tophat (whitehat) morphological operator to find light
# regions against a dark background 
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

# test tophat output
# cv2.imshow("Tophat output", tophat)
# cv2.waitKey(0)

# compute the Scharr gradient of the tophat image, then scale
# the rest back into the range [0, 255]
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")

# test gradX output
# cv2.imshow("GradX output", gradX)
# cv2.waitKey(0)

# apply a closing operation using the rectangular kernel to help
# cloes gaps in between data
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)

# test gradX output
# cv2.imshow("GradX output", gradX)
# cv2.waitKey(0)

# then apply Otsu's thresholding method to binarize the image
thresh = cv2.threshold(gradX, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# test thresh output
# cv2.imshow("Thresh output", thresh)
# cv2.waitKey(0)

# apply a second closing operation to the binary image, again
# to help close gaps between ktm data regions
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

# test thresh output
# cv2.imshow("Thresh output", thresh)
# cv2.waitKey(0)

# find contours in the thresholded image, then initialize the
# list of digit locations
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
data = []

# loop over the contours
cnts_rev = list(reversed(cnts))
for (i, c) in enumerate(cnts_rev):
	(x, y, w, h) = cv2.boundingRect(c)
	x = x - 5
	y = y - 5
	w = w + 8
	h = h + 8
	ar = w / float(h)
	# check for height and width
	if w > 80 and h < 25:
		# crop selected region and change color to gray
		cropped = image[y:y+h, x:x+w]
		gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

		# scale image and doing image processing
		scale_x = 5
		block = cv2.resize(gray, (int(gray.shape[1] * scale_x), int(gray.shape[0] * scale_x)))
		block = cv2.erode(block,(9,9))
		block = cv2.GaussianBlur(block, (3, 3), 0)

		# change data format into array and filter it
		pil_img = Image.fromarray(block)
		block_pil = pil_img.filter(ImageFilter.SHARPEN)

		# convert block pil to image with pytesseract
		ocr = pytesseract.image_to_string(block_pil)
		data.append(ocr)

		# display the change
		print x, y, w, h, '\n'
		cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)
		cv2.putText(image, ocr, (x+w+10, y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
		cv2.imshow("img", image)
		cv2.waitKey(0)   

# optional, to pring data in terminal 
data_npm = ["NPM", "NAMA", "FAKULTAS", "JURUSAN"]
data_output = list(reversed(data))
for i, data in enumerate(data_output):
	if i < 2:
		print data_npm[i], '\t\t:', data
	elif i >= 2 and i <= 3:	
		print data_npm[i], '\t:', data