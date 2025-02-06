import cv2

#membaca gambar dalam graycale
image = cv2.imread('photo.jpg', 0)

#menerapkan adptive thresholding
adaptive_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#menampilkan hasil
cv2.imshow('Adaptive Thresholding', adaptive_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows