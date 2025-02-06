import cv2 
import numpy as np

#baca citra dalam grayscale
image = cv2.imread('photo.jpg', cv2.IMREAD_GRAYSCALE)

#terapkan thresholding global
ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

#tampilakn hasil segmentasi
cv2.imshow('Threshold Image', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()