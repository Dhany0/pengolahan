import cv2
import numpy as np 
from skimage.feature import local_binary_pattern

#membaca gambar dalam grayscale
image = cv2.imread('photo.jpeg', 0)

#menerapkan local binary patten
radius = 1
n_points = 8 * radius
lbp = local_binary_pattern(image, n_points, radius, method='uniform')

#menampilkan hasil
cv2.imshow('local binary Pattern', lbp.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()