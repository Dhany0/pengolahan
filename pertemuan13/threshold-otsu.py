import cv2

#baca citra dalam grayscale
image = cv2.imread('photo.jpg', cv2.IMREAD_GRAYSCALE)

#terapkan otsu's thresholding
ret, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#tampilakn hasil segmentasi
cv2.imread('Otsu Thresholding', otsu_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()