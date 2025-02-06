import cv2
import numpy as np
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore

#inisialisasi model ccn
model = Sequential()

#tambahkan kompolutianal layer lainnya
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

#tambahkan konvolutional laiinnya
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#flattening
model.add(Flatten())

#fully conneted layer
model.add(Dense(128, activation='relu'))
model.add(Dense(9, activation='softmax'))

#comple model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

X_train = np.random.rand(100, 64, 64, 3) 
y_train = np.random.randint(10, size=(100,))  

X_test = np.random.rand(20, 64, 64, 3) 
y_test = np.random.randint(10, size=(20,)) 

#latih model
model.fit(X_train, to_categorical(y_train), epochs=10, batch_size=32)

#evaluasi model
loss, accuracy = model.evaluate(X_test, to_categorical(y_test))
print(f'Akurasi CNN: {accuracy * 100:2f}%')