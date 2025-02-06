import cv2 
# Membaca gambar 
image = cv2.imread('photo.jpg') 
# Mengecilkan ukuran gambar menjadi setengahnya
resized_image = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)
# Menampilkan gambar yang diubah ukurannya
cv2.imshow('Resized Image', resized_image) 
# Menunggu hingga ada input dari keyboard 
cv2.waitKey(0) 
# Menutup semua jendela 
cv2.destroyAllWindows() 