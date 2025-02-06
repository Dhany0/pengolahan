import cv2

#baca citra dengan graycale
image = cv2.imread('photo.jpg', cv2.IMREAD_GRAYSCALE)

#terapkan deteksi tepi dengna metode canny
edges = cv2.Canny(image, 100, 200)

#tampilkan hasil dekteksi tepi
cv2.imshow('Harris Corner Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()