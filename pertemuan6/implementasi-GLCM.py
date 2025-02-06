from skimage.feature import graycomatrix, graycoprops
import cv2

#membaca gambar dalam greycale
image = cv2.imread('photo.jpeg', 0)


#menghitung GLCM
glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

#menghitung pitur tektur dari glcm
contrast = graycoprops(glcm, 'contrast')[0, 0]
energy = graycoprops(glcm, 'energy') [0, 0]

print(f'Contrast: {contrast}, {energy}')
