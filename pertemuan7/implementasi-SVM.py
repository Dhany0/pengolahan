import cv2 
import numpy as np 
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#membaca dataset citra
#x adalah array fitur dan y adalah array label (kelas)
X = [[1, 2], [3, 4], [5, 6], [7, 8]] #fitur dari citra
y = [0, 1, 0, 1] #lebel dari citra

#split data untuk prlatihan  daa pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

#inisial model SVM
clf = svm.SVC(kernel='linear')

#latih model menggunakan data pelatihan
clf.fit(X_train, y_train)

#prediksi mengunakan data uji
y_pred = clf.predict(X_test)

#hitung akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi: {accuracy * 100:2f}%')