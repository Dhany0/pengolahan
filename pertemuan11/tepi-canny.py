import cv2

#baca citra grayscale
image = cv2.imread('photo.jpg', cv2.IMREAD_GRAYSCALE)

#deteksi tepi menggunakan canny
edges = cv2.Canny(image, 100, 200)

#tampilkan hasil
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()