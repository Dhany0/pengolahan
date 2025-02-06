import cv2  

#membaca gambar
image = cv2.imread('photo.jpeg')

#inisialisasi objec SIFT
sift = cv2.SIFT_create()

#mendeteksi keypoints dan komputasi deskriptor
keypoints, descriptors = sift.detectAndCompute(image, None)

#menggambar keypoinst di cintra
sift_image = cv2.drawKeypoints(image, keypoints, None)

#menampilka hasil
cv2.imshow('SIFT Features', sift_image)
cv2.waitKey(0)
cv2.destroyAllWindows