import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QInputDialog, QMessageBox,
                            QProgressDialog)
from PyQt5.QtGui import QPainter, QPen, QImage, QPixmap
from PyQt5.QtCore import Qt, QPoint, QTimer
import tensorflow as tf
import os
from scipy import ndimage
import datetime
from PIL import Image
import glob
from tqdm import tqdm

# Включаем eager execution
tf.config.run_functions_eagerly(True)

# Добавляем класс для мониторинга прогресса обучения
class TrainingMonitor(tf.keras.callbacks.Callback):
    def __init__(self, progress_dialog, total_epochs):
        super().__init__()
        self.progress_dialog = progress_dialog
        self.total_epochs = total_epochs
        
    def on_epoch_begin(self, epoch, logs=None):
        self.progress_dialog.setLabelText(f"Эпоха {epoch+1}/{self.total_epochs}")
        
    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get('accuracy', 0) * 100
        self.progress_dialog.setValue(epoch + 1)
        self.progress_dialog.setLabelText(f"Эпоха {epoch+1}/{self.total_epochs} завершена. Точность: {acc:.2f}%")


class DrawingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(280, 280)
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.drawing = False
        self.last_point = QPoint()
        self.main_window = None
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()
    
    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.black, 20, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            if self.main_window:
                self.main_window.predict_digit()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)
    
    def clear(self):
        self.image.fill(Qt.white)
        self.update()
        if self.main_window:
            self.main_window.result_label.setText("Результат: -")


class DigitRecognizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Распознавание рукописных цифр")
        self.setGeometry(300, 300, 500, 350)
        
        # Загрузка обученной модели
        try:
            # Пробуем разные форматы файлов
            if os.path.exists('digit_model.keras'):
                self.model = tf.keras.models.load_model('digit_model.keras')
                self.model_format = '.keras'
                print("Модель успешно загружена (.keras)")
            elif os.path.exists('digit_model.h5'):
                self.model = tf.keras.models.load_model('digit_model.h5')
                self.model_format = '.h5'
                print("Модель успешно загружена (.h5)")
            else:
                print("Файл модели не найден! Сначала запустите train_model.py")
                sys.exit(1)
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            sys.exit(1)
        
        # Главный виджет и компоновка
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        
        # Виджет для рисования
        self.drawing_widget = DrawingWidget(main_widget)
        self.drawing_widget.main_window = self
        main_layout.addWidget(self.drawing_widget)
        
        # Панель управления
        control_layout = QVBoxLayout()
        
        # Метка результата
        self.result_label = QLabel("Результат: -")
        self.result_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        control_layout.addWidget(self.result_label)
        
        # Кнопка очистки
        clear_button = QPushButton("Очистить")
        clear_button.clicked.connect(self.drawing_widget.clear)
        control_layout.addWidget(clear_button)
        
        # Добавьте кнопку "Сохранить пример" в интерфейс
        save_button = QPushButton("Сохранить пример")
        save_button.clicked.connect(self.save_example)
        control_layout.addWidget(save_button)
        
        # Добавьте кнопку "Создать датасет для обучения" в интерфейс
        dataset_button = QPushButton("Создать датасет для обучения")
        dataset_button.clicked.connect(self.save_examples_to_dataset)
        control_layout.addWidget(dataset_button)
        
        # Добавляем кнопку дообучения модели
        retrain_button = QPushButton("Дообучить модель")
        retrain_button.clicked.connect(self.retrain_model)
        control_layout.addWidget(retrain_button)
        
        # Добавляем панель управления в главную компоновку
        main_layout.addLayout(control_layout)
        
        self.setCentralWidget(main_widget)
    
    def predict_digit(self):
        # Получение изображения из виджета рисования
        img = self.drawing_widget.image.copy().scaled(28, 28, Qt.IgnoreAspectRatio)
        
        # Конвертация в numpy массив
        buffer = img.bits().asstring(img.byteCount())
        img_array = np.frombuffer(buffer, dtype=np.uint8).reshape((28, 28, 4))
        
        # Извлечение только канала яркости и инвертирование цветов
        img_array = 255 - img_array[:, :, 0]
        
        # УЛУЧШЕНИЕ: Центрирование изображения цифры
        # Находим центр масс изображения
        cy, cx = ndimage.center_of_mass(img_array)
        
        # Смещение для центрирования
        rows, cols = img_array.shape
        shift_x = np.round(cols/2 - cx).astype(int)
        shift_y = np.round(rows/2 - cy).astype(int)
        
        # Применяем смещение
        img_array = ndimage.shift(img_array, (shift_y, shift_x), mode='constant')
        
        # Нормализация и преобразование формы
        img_array = img_array.reshape(1, 28, 28, 1).astype('float32') / 255.0
        
        # Предсказание
        prediction = self.model.predict(img_array, verbose=0)  # Убираем лишний вывод
        digit = np.argmax(prediction)
        confidence = prediction[0][digit] * 100
        
        # Вывод результата
        self.result_label.setText(f"Результат: {digit} ({confidence:.1f}%)")

    def save_example(self):
        # Создаем директорию, если её нет
        if not os.path.exists('examples'):
            os.makedirs('examples')
        
        # Получаем текущую цифру из текста
        current_text = self.result_label.text()
        if "Результат: -" in current_text:
            return
        
        predicted_digit = current_text.split()[1][0]  # Извлекаем первый символ после "Результат: "
        
        # Запрос ввода правильной цифры
        true_digit, ok = QInputDialog.getText(
            self, 
            'Ввод правильной цифры', 
            f'Нейросеть распознала как "{predicted_digit}". Введите правильную цифру:',
            text=predicted_digit  # По умолчанию предлагаем распознанное значение
        )
        
        # Если пользователь нажал 'Отмена' или оставил поле пустым
        if not ok or not true_digit:
            return
        
        # Проверяем, что введена только одна цифра от 0 до 9
        if not (true_digit.isdigit() and len(true_digit) == 1):
            QMessageBox.warning(self, "Ошибка", "Пожалуйста, введите одну цифру от 0 до 9.")
            return
        
        # Генерируем имя файла с текущей датой и временем
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Добавляем информацию о предсказании в имя файла
        filename = f"examples/digit_{true_digit}_pred{predicted_digit}_{timestamp}.png"
        
        # Сохраняем изображение
        self.drawing_widget.image.save(filename)
        print(f"Пример сохранен как {filename} (распознано как {predicted_digit}, правильно {true_digit})")

    def save_examples_to_dataset(self):
        # Добавьте эту функцию в класс DigitRecognizerApp
        from PyQt5.QtWidgets import QMessageBox
        
        # Проверяем наличие сохраненных примеров
        if not os.path.exists('examples') or not os.listdir('examples'):
            QMessageBox.information(self, "Информация", "Нет сохраненных примеров для создания датасета.")
            return
        
        # Создаем директорию для датасета
        if not os.path.exists('custom_dataset'):
            os.makedirs('custom_dataset')
        
        # Обрабатываем все сохраненные примеры
        import shutil
        
        examples = glob.glob('examples/*.png')
        for idx, example in enumerate(examples):
            # Извлекаем правильную цифру из имени файла
            filename = os.path.basename(example)
            true_digit = filename.split('_')[1]
            
            # Копируем файл в датасет с новым именем
            new_filename = f"custom_dataset/{true_digit}_{idx}.png"
            shutil.copy(example, new_filename)
        
        QMessageBox.information(
            self, 
            "Датасет создан", 
            f"Датасет успешно создан в папке 'custom_dataset' ({len(examples)} примеров)."
        )

    def retrain_model(self):
        """Функция для дообучения модели на кастомном датасете."""
        # Проверяем наличие датасета
        if not os.path.exists('custom_dataset') or len(glob.glob('custom_dataset/*.png')) == 0:
            QMessageBox.information(
                self, 
                "Информация", 
                "Нет данных для дообучения. Сначала создайте датасет."
            )
            return
        
        # Явно включаем eager execution
        tf.config.run_functions_eagerly(True)
        
        # Подготовка данных из custom_dataset
        x_custom = []
        y_custom = []
        
        # Показываем прогресс загрузки данных
        loading_dialog = QProgressDialog("Загрузка данных...", None, 0, 0, self)
        loading_dialog.setWindowTitle("Подготовка к обучению")
        loading_dialog.setModal(True)
        loading_dialog.setCancelButton(None)
        loading_dialog.setMinimumDuration(0)
        loading_dialog.show()
        QApplication.processEvents()
        
        files = glob.glob('custom_dataset/*.png')
        loading_dialog.setMaximum(len(files))
        
        for i, img_path in enumerate(files):
            # Показываем прогресс загрузки
            loading_dialog.setValue(i)
            loading_dialog.setLabelText(f"Загрузка изображений {i+1}/{len(files)}")
            QApplication.processEvents()
            
            # Получаем правильный ответ из имени файла
            true_digit = int(os.path.basename(img_path).split('_')[0])
            
            # Обрабатываем изображение
            img = Image.open(img_path).convert('L').resize((28, 28))
            img_array = np.array(img).reshape(28, 28, 1).astype('float32') / 255.0
            
            x_custom.append(img_array)
            y_custom.append(true_digit)
        
        loading_dialog.close()
        
        # Преобразуем в массивы numpy
        x_custom = np.array(x_custom)
        
        # Используем другой метод для преобразования в категориальный формат
        y_custom_categorical = np.zeros((len(y_custom), 10), dtype='float32')
        for i, label in enumerate(y_custom):
            y_custom_categorical[i, label] = 1.0
        
        # Создаем диалог прогресса для отображения процесса обучения
        epochs = 10
        progress_dialog = QProgressDialog("Обучение модели...", "Отмена", 0, epochs, self)
        progress_dialog.setWindowTitle("Дообучение модели")
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.show()
        
        # Создаем монитор обучения с нашим dialog
        monitor = TrainingMonitor(progress_dialog, epochs)
        
        try:
            # Отключаем все кнопки интерфейса на время обучения
            for btn in self.findChildren(QPushButton):
                btn.setEnabled(False)
            
            # ИЗМЕНЕНО: Загружаем модель из файла с учетом формата
            model_path = f'digit_model{self.model_format}'
            base_model = tf.keras.models.load_model(model_path, compile=False)
            
            # ИЗМЕНЕНО: Перекомпилируем модель с новым оптимизатором
            base_model.compile(
                loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
            )
            
            batch_size = max(1, min(8, len(x_custom) // 2))
            print(f"Начинаем дообучение на {len(x_custom)} примерах, размер батча: {batch_size}")
            
            # Используем tqdm для отображения прогресса в консоли
            with tqdm(total=epochs, desc="Обучение") as pbar:
                # Создаем callback для обновления tqdm
                class TqdmCallback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        pbar.update(1)
                        pbar.set_postfix({
                            'loss': f"{logs.get('loss', 0):.4f}",
                            'accuracy': f"{logs.get('accuracy', 0):.4f}"
                        })
                
                # Обучаем модель с двумя callbacks
                history = base_model.fit(
                    x_custom, y_custom_categorical, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    verbose=0,  # Отключаем встроенный вывод прогресса
                    callbacks=[monitor, TqdmCallback()]
                )
                
                # Сохраняем модель в том же формате
                try:
                    base_model.save(model_path)
                    print(f"Модель сохранена как '{model_path}'")
                except Exception as save_error:
                    print(f"Ошибка при сохранении модели: {save_error}")
                    # Пробуем альтернативные форматы
                    if self.model_format == '.keras':
                        base_model.save('digit_model.h5')
                        self.model_format = '.h5'
                        print("Модель сохранена как 'digit_model.h5'")
                    else:
                        base_model.save('digit_model.keras')
                        self.model_format = '.keras'
                        print("Модель сохранена как 'digit_model.keras'")
            
            # Включаем кнопки обратно
            for btn in self.findChildren(QPushButton):
                btn.setEnabled(True)
            
            QMessageBox.information(
                self, 
                "Дообучение завершено", 
                f"Модель успешно дообучена на {len(x_custom)} пользовательских примерах и сохранена."
            )
        except Exception as e:
            # Обработка возможных ошибок
            QMessageBox.critical(
                self,
                "Ошибка при дообучении",
                f"Произошла ошибка при дообучении модели:\n{str(e)}"
            )
            print(f"Ошибка при дообучении: {e}")
        finally:
            # Закрываем диалог прогресса
            progress_dialog.close()
            
            # Включаем кнопки обратно в любом случае
            for btn in self.findChildren(QPushButton):
                btn.setEnabled(True)


if __name__ == "__main__":
    # Скрываем некоторые предупреждения TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    app = QApplication(sys.argv)
    window = DigitRecognizerApp()
    window.show()
    sys.exit(app.exec_()) 