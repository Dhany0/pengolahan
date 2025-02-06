import cv2 
import numpy as np

#baca citra 
image = cv2.imread('photo.jpg')

#konversi citra ke grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#terapakn thresholding
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#penghapusan noise
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)

#tentukan area latar belakang
sure_bg = cv2.dilate(opening, kernel, iterations=3)

#tentukan area objek
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255,0)

#temukan area perbatasan
sure_fg = np.uint8(sure_fg)
unkown = cv2.subtract(sure_bg, sure_fg)

#marker labeling
ret, markers = cv2.connectedComponents(sure_fg)

#tampilkan satu ke semua marker sehingga bacround akan menjadi 1 bukan 0
markers = markers + 1

#tandai area perbatasan dengan 0
markers[unkown == 255] = 0

#terapkan watershed
markers = cv2.watershed(image, markers)
image[markers == -1] = [0, 0, 255]

#tampilkan hasil
cv2.imshow('Watershed Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()