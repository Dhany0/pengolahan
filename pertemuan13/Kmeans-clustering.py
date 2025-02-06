import cv2
import numpy as np

#baca citra
image = cv2.imread('photo.jpg')

#ubah format citra ken dalam satu demensi
Z = image.reshape((-1,3))

#ubah tife data ke float32
Z = np.float32(Z)

#tentukan kreteria dan jumlah cluster
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3 #misalnya, 3 cluster

#terapkan kmeans clustering
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

#konversi hasil kembali ke format citra
center = np.uint8(center)
res = center[label.flatten()]
segmented_image = res.reshape((image.shape))

#tmapilkan hasil segmentasi
cv2.imshow('Segmented_Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()