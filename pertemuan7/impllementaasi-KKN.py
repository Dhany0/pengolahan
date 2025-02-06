from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Menambahkan lebih banyak data agar pembagian menjadi mungkin
X = [[1, 2], [3, 4], [5, 6], [7, 8]]
y = [0, 1, 0, 1]

# Split data untuk pelatihan (menyesuaikan test_size untuk dataset kecil)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Inisialisasi model KNN
knn = KNeighborsClassifier(n_neighbors=2)

# Latih model menggunakan data pelatihan
knn.fit(X_train, y_train)

# Prediksi menggunakan data uji
y_pred = knn.predict(X_test)

# Hitung akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi KNN: {accuracy * 100:.2f}%')
