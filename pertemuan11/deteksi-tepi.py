import cv2 
import numpy as np 

#baca citra grayscale
image = cv2.imread('photo.jpg', cv2.IMREAD_GRAYSCALE)

#deteksi tepi menggunakan sobel
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5) #sobel X
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

#tampilkan hasil
cv2.imshow('sobel X', sobelx)
cv2.imshow('sobel Y', sobely)
cv2.waitKey(0)
cv2.destroyAllWindows()