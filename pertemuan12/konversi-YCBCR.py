import cv2

#baca citra dalam format rgb
image = cv2.imread('photo.jpg')

#konversi citra dari rgb ke ycbcr
ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

#ekstrak cannel y
Y_channel = ycbcr_image[:, :, 0]

#tampilkan channel Y
cv2.imshow('Y Channel', Y_channel)
cv2.waitKey(0)
cv2.destroyAllWindows()