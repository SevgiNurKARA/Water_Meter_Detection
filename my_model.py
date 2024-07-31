import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import numpy as np

# Veri seti yolunu belirleyin
data_dir = 'data_dir'

# Eğitim, doğrulama ve test klasörlerini oluşturun
train_dir = 'train'
val_dir = 'val'
test_dir = 'test'

for dir in [train_dir, val_dir, test_dir]:
    for category in ['water_meter', 'not_water_meter']:
        os.makedirs(os.path.join(dir, category), exist_ok=True)

# Veri setini bölme
for category in ['water_meter', 'not_water_meter']:
    files = os.listdir(os.path.join(data_dir, category))
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=0.2, random_state=42)

    for file in train_files:
        shutil.copy(os.path.join(data_dir, category, file), os.path.join(train_dir, category, file))
    for file in val_files:
        shutil.copy(os.path.join(data_dir, category, file), os.path.join(val_dir, category, file))
    for file in test_files:
        shutil.copy(os.path.join(data_dir, category, file), os.path.join(test_dir, category, file))

# Veri artırma ve ön işleme
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1. / 255)

# Eğitim ve doğrulama veri setlerini oluşturun
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Sınıf ağırlıklarını hesaplayın
train_labels = np.array([0] * len(os.listdir(os.path.join(train_dir, 'not_water_meter'))) + 
                        [1] * len(os.listdir(os.path.join(train_dir, 'water_meter'))))

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

class_weights = {0: class_weights[0], 1: class_weights[1]}

# Model oluşturma
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Model derleme
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modeli eğitme
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32,
    class_weight=class_weights
)

# Modeli kaydetme
model.save('my_model.keras')
