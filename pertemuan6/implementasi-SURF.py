import cv2

# Membaca gambar
image = cv2.imread('photo.jpeg')

# Inisialisasi objek SURF
surf = cv2.xfeatures2d.SURF_create()

# Mendeteksi keypoints dan komputasi deskriptor
keypoints, descriptors = surf.detectAndCompute(image, None)

# Menggambar keypoints di citra
surf_image = cv2.drawKeypoints(image, keypoints, None)

# Menampilkan hasil
cv2.imshow('SURF Feature', surf_image)
cv2.waitKey(0)
cv2.destroyAllWindows() 
