import cv2 
import numpy as np

#membaca gambar dalam grayscal
image = cv2.imread('photo.jpg')

#menerapkan thresholding
ret, thresh_image = cv2.threshold(image,127, 255, cv2.THRESH_BINARY)

#menampilkan hasil
cv2.imshow('Thresholded Image', thresh_image)
cv2.waitKey(0)
cv2.destroyAllWindows(0)