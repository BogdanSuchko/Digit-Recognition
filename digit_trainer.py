import sys
import os
import numpy as np
import glob
import datetime
from tqdm import tqdm
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QSpinBox, 
                           QProgressDialog, QMessageBox)
from PyQt5.QtGui import QPainter, QPen, QImage, QPixmap
from PyQt5.QtCore import Qt, QPoint, QSize
import tensorflow as tf
from PIL import Image
from scipy import ndimage

# Включаем eager execution
tf.config.run_functions_eagerly(True)

class DrawingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(280, 280)
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.drawing = False
        self.last_point = QPoint()
        
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
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)
    
    def clear(self):
        self.image.fill(Qt.white)
        self.update()


class DigitTrainerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Тренер цифр для нейросети")
        self.setGeometry(300, 300, 600, 400)
        
        # Определение формата модели
        self.model_format = '.h5'
        if os.path.exists('digit_model.keras'):
            self.model_format = '.keras'
        
        # Создаем основной виджет и его компоновку
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        
        # Левая панель с настройками
        left_panel = QVBoxLayout()
        
        # Выбор цифры для обучения
        digit_layout = QHBoxLayout()
        digit_layout.addWidget(QLabel("Выберите цифру для обучения:"))
        self.digit_selector = QSpinBox()
        self.digit_selector.setRange(0, 9)
        self.digit_selector.setValue(4)  # По умолчанию 4, как в запросе
        self.digit_selector.setFixedHeight(30)
        digit_layout.addWidget(self.digit_selector)
        left_panel.addLayout(digit_layout)
        
        # Текущая статистика
        self.stats_label = QLabel("Нарисовано: 0")
        left_panel.addWidget(self.stats_label)
        
        # Информация о процессе
        info_text = (
            "1. Выберите цифру для обучения\n"
            "2. Нарисуйте эту цифру справа\n"
            "3. Нажмите 'Сохранить' после каждого рисунка\n"
            "4. Повторите несколько раз\n"
            "5. Нажмите 'Дообучить модель'"
        )
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        left_panel.addWidget(info_label)
        
        # Кнопки управления
        save_button = QPushButton("Сохранить нарисованную цифру")
        save_button.clicked.connect(self.save_digit)
        left_panel.addWidget(save_button)
        
        clear_button = QPushButton("Очистить")
        clear_button.clicked.connect(self.clear_drawing)
        left_panel.addWidget(clear_button)
        
        train_button = QPushButton("Дообучить модель")
        train_button.clicked.connect(self.train_model)
        left_panel.addWidget(train_button)
        
        # Добавляем левую панель в основную компоновку
        main_layout.addLayout(left_panel, 1)
        
        # Правая панель с областью для рисования
        right_panel = QVBoxLayout()
        
        # Метка для инструкции
        draw_label = QLabel(f"Нарисуйте цифру {self.digit_selector.value()}")
        draw_label.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(draw_label)
        
        # Область для рисования
        self.drawing_widget = DrawingWidget()
        right_panel.addWidget(self.drawing_widget)
        
        # Обновляем метку при изменении выбранной цифры
        self.digit_selector.valueChanged.connect(
            lambda val: draw_label.setText(f"Нарисуйте цифру {val}")
        )
        
        # Добавляем правую панель в основную компоновку
        main_layout.addLayout(right_panel, 2)
        
        self.setCentralWidget(main_widget)
        
        # Счетчик нарисованных цифр
        self.drawn_count = 0
        
        # Создаем директории, если их нет
        if not os.path.exists('examples'):
            os.makedirs('examples')
        if not os.path.exists('custom_dataset'):
            os.makedirs('custom_dataset')
    
    def save_digit(self):
        # Получаем текущую выбранную цифру
        digit = self.digit_selector.value()
        
        # Генерируем имя файла с временной меткой
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        examples_filename = f"examples/digit_{digit}_train_{timestamp}.png"
        dataset_filename = f"custom_dataset/{digit}_{self.drawn_count}.png"
        
        # Сохраняем изображение в обе папки
        self.drawing_widget.image.save(examples_filename)
        self.drawing_widget.image.save(dataset_filename)
        
        # Увеличиваем счетчик
        self.drawn_count += 1
        self.stats_label.setText(f"Нарисовано: {self.drawn_count}")
        
        # Очищаем область для рисования для следующего примера
        self.drawing_widget.clear()
        
        print(f"Сохранено: {examples_filename} и {dataset_filename}")
    
    def clear_drawing(self):
        self.drawing_widget.clear()
    
    def train_model(self):
        # Проверяем, были ли нарисованы цифры
        if self.drawn_count == 0:
            QMessageBox.warning(
                self, 
                "Предупреждение", 
                "Вы не нарисовали ни одной цифры. Нарисуйте несколько примеров перед обучением."
            )
            return
        
        # Спрашиваем пользователя о подтверждении
        confirm = QMessageBox.question(
            self,
            "Подтверждение",
            f"Вы нарисовали {self.drawn_count} примеров. Хотите дообучить модель на этих данных?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if confirm == QMessageBox.No:
            return
        
        # Отключаем кнопки на время обучения
        for btn in self.findChildren(QPushButton):
            btn.setEnabled(False)
        
        # Загружаем данные из custom_dataset
        self.load_and_train_model()
        
        # Включаем кнопки после обучения
        for btn in self.findChildren(QPushButton):
            btn.setEnabled(True)
    
    def load_and_train_model(self):
        # Показываем прогресс загрузки данных
        loading_dialog = QProgressDialog("Загрузка данных...", None, 0, 0, self)
        loading_dialog.setWindowTitle("Подготовка к обучению")
        loading_dialog.setModal(True)
        loading_dialog.setCancelButton(None)
        loading_dialog.setMinimumDuration(0)
        loading_dialog.show()
        QApplication.processEvents()
        
        # Подготовка данных
        x_custom = []
        y_custom = []
        
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
        
        # Загружаем модель для дообучения
        try:
            model_path = f'digit_model{self.model_format}'
            print(f"Загрузка модели из {model_path}")
            
            base_model = tf.keras.models.load_model(model_path, compile=False)
            
            # Перекомпилируем модель
            base_model.compile(
                loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
            )
            
            # Настройки для обучения
            epochs = 10
            batch_size = max(1, min(8, len(x_custom) // 2))
            
            # Диалог прогресса обучения
            progress_dialog = QProgressDialog("Обучение модели...", None, 0, epochs, self)
            progress_dialog.setWindowTitle("Дообучение модели")
            progress_dialog.setMinimumDuration(0)
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.show()
            
            # Создаем монитор для отображения прогресса
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
                    self.progress_dialog.setLabelText(
                        f"Эпоха {epoch+1}/{self.total_epochs} завершена. Точность: {acc:.2f}%"
                    )
            
            monitor = TrainingMonitor(progress_dialog, epochs)
            
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
                
                # Обучаем модель
                history = base_model.fit(
                    x_custom, y_custom_categorical, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    verbose=0,
                    callbacks=[monitor, TqdmCallback()]
                )
                
                # Сохраняем обновленную модель
                try:
                    base_model.save(model_path)
                    print(f"Модель сохранена как {model_path}")
                except Exception as e:
                    print(f"Ошибка при сохранении модели: {e}")
                    # Пробуем альтернативный формат
                    if self.model_format == '.keras':
                        base_model.save('digit_model.h5')
                        print("Модель сохранена как 'digit_model.h5'")
                    else:
                        base_model.save('digit_model.keras')
                        print("Модель сохранена как 'digit_model.keras'")
            
            progress_dialog.close()
            
            # Информируем пользователя об успешном обучении
            QMessageBox.information(
                self,
                "Обучение завершено",
                f"Модель успешно дообучена на {len(x_custom)} примерах."
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Ошибка при обучении",
                f"Произошла ошибка при дообучении модели:\n{str(e)}"
            )
            print(f"Ошибка при дообучении: {e}")


if __name__ == "__main__":
    # Скрываем некоторые предупреждения TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    app = QApplication(sys.argv)
    window = DigitTrainerApp()
    window.show()
    sys.exit(app.exec_()) 