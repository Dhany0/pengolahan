import cv2
import numpy as np

#membaca gambar dalam grayscale
image = cv2.imread('photo.jpeg', 0)

#konversi gambar ke tife fload32
gray = np.float32(image)

#menerapkan haris corner detektor
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

#dilasi untuk menonjolkan sudut yang terdeteksi
dst = cv2.dilate(dst, None)

# Konversi gambar grayscale ke warna (RGB)
image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

#thresholding untuk menandai sudut
image_color[dst > 0.01 * dst.max()] = [0, 0, 255]

#menamplkan hasil
cv2.imshow('Harris Corner', image)
cv2.waitKey(0)
cv2.destroyAllWindows