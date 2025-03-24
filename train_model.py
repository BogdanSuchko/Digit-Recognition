import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Загрузка данных MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Предобработка данных
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
x_train /= 255
x_test /= 255

# Преобразование меток в категориальный формат
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Создание модели CNN
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Компиляция модели
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Аугментация данных
datagen = ImageDataGenerator(
    rotation_range=10,  # Небольшие повороты
    width_shift_range=0.1,  # Небольшие сдвиги по горизонтали
    height_shift_range=0.1,  # Небольшие сдвиги по вертикали
    zoom_range=0.1  # Небольшие изменения масштаба
)

# Обучение с аугментацией
model.fit(datagen.flow(x_train, y_train, batch_size=max(1, min(8, len(x_train) // 2))),
          epochs=15,  # Увеличиваем количество эпох
          validation_data=(x_test, y_test))

# Сохранение модели
try:
    # Пробуем сохранить в формате .keras
    model.save('digit_model.keras')
    print("Модель сохранена как 'digit_model.keras'")
except Exception as e:
    # Если не получается, используем формат .h5
    print(f"Не удалось сохранить в формате .keras: {e}")
    model.save('digit_model.h5')
    print("Модель сохранена как 'digit_model.h5'") 