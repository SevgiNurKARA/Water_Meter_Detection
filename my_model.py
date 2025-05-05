import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMG_SIZE = 224  
BATCH_SIZE = 32
EPOCHS = 20  
LEARNING_RATE = 0.0001
N_FOLDS = 3

def create_model():

    # Pre-trained ResNet50 modelini yükle 
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Özellik çıkarma kısmını donduralım 
    for layer in base_model.layers:
        layer.trainable = False
    
    # Model üzerine kendi sınıflandırıcımı ekledm
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Regularization: Dropout ve Batch Normalization
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # İkili sınıflandırma için çıkış katmanı (Su sayacı var/yok)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Model derleme
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy', 
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def extract_features(img):
 
    features = {}
    
    try:
       
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # 1. Mavi renk tespiti - basitleştirilmiş
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_ratio = np.sum(blue_mask > 0) / (img.shape[0] * img.shape[1])
        features['blue_ratio'] = blue_ratio
        
        # 2. Temel görüntü özellikleri
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features['brightness'] = np.mean(gray)
        features['contrast'] = np.std(gray)
        
        # 3. Basitleştirilmiş kenar tespiti
        edges = cv2.Canny(gray, 100, 200)
        features['edge_density'] = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
        
        # 4. Dairesel şekil tespiti - ResNet50 için eklendi
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
            param1=50, param2=30, minRadius=10, maxRadius=min(100, min(img.shape[0], img.shape[1])//2)
        )
        features['circle_count'] = 0 if circles is None else len(circles[0])
        
    except Exception as e:
        print(f"extract_features fonksiyonunda hata: {e}")
        features = {
            'blue_ratio': 0, 
            'brightness': 0, 
            'contrast': 0, 
            'edge_density': 0,
            'circle_count': 0
        }
    
    return features

def load_all_images(data_dir, classes, max_per_class=1000):

    images = []
    labels = []
    image_paths = []
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        class_label = 1 if class_name == 'water_meter' else 0
        
        print(f"'{class_name}' sınıfından görüntüler yükleniyor...")
        
        try:
            img_files = os.listdir(class_dir)
            img_files = img_files[:max_per_class]
            
            for img_name in img_files:
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                img_path = os.path.join(class_dir, img_name)
                image_paths.append(img_path)
                labels.append(class_label)
        except Exception as e:
            print(f"HATA: {class_dir} dizininden görüntü yüklenirken: {e}")
    
    labels = np.array(labels)
    
    return image_paths, labels

def augment_data_generator():

    train_datagen = ImageDataGenerator(
        rescale=1./255,  # Normalizasyon
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    validation_datagen = ImageDataGenerator(rescale=1./255) 
    
    return train_datagen, validation_datagen

def fine_tune_model(model, num_unfrozen_layers=10):
    """
    ResNet50'nin son katmanlarını eğitim için açar (fine-tuning)
    """
    # Başlangıçta tüm katmanları dondur
    for layer in model.layers:
        layer.trainable = False
    
    # Son num_unfrozen_layers katmanı eğitim için aç
    for layer in model.layers[-num_unfrozen_layers:]:
        layer.trainable = True
    
    # Model derleme
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE/10),
        loss='binary_crossentropy', 
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def create_generator_from_paths(image_paths, labels, datagen):

    # Görüntü yükleme ve önişleme fonksiyonu
    def load_and_preprocess_image(path):
        try:
            img = tf.io.read_file(path)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
            img = img / 255.0  # Normalizasyon
            return img
        except:
            return tf.zeros([IMG_SIZE, IMG_SIZE, 3])
    
    # Veri setini oluştur
    paths_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    images_ds = paths_ds.map(
        load_and_preprocess_image, 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((images_ds, labels_ds))
    
    # Karıştır ve batch'le
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def train_with_kfold(image_paths, labels, output_dir):
    # Stratified K-Fold oluştur
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    # Model performans metriklerini saklamak için
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels)):
        print(f"\n{'='*20} Fold {fold+1}/{N_FOLDS} {'='*20}")
        
        # Eğitim ve doğrulama setlerini oluştur
        train_paths = [image_paths[i] for i in train_idx]
        train_labels = labels[train_idx]
        val_paths = [image_paths[i] for i in val_idx]
        val_labels = labels[val_idx]
        
        # Veri jeneratörleri
        train_datagen, val_datagen = augment_data_generator()
        
        # Eğitim jeneratörü
        train_generator = create_generator_from_paths(train_paths, train_labels, train_datagen)
        
        # Doğrulama jeneratörü
        val_generator = create_generator_from_paths(val_paths, val_labels, val_datagen)
        
        # Model oluştur
        model = create_model()
        
        # Callbacks
        model_save_path = os.path.join(output_dir, f'water_meter_model_fold_{fold+1}.keras')
        callbacks = [
            ModelCheckpoint(
                model_save_path, monitor='val_accuracy', 
                save_best_only=True, mode='max', verbose=1
            ),
            EarlyStopping(
                monitor='val_loss', patience=4,
                restore_best_weights=True, verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss', factor=0.2,
                patience=2, min_lr=1e-6, verbose=1
            )
        ]
        
        # Model eğitimi
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=val_generator,
            callbacks=callbacks
        )
        
        # Fine-tuning
        model = fine_tune_model(model)
        
        history_fine = model.fit(
            train_generator,
            epochs=8,  # Fine-tuning için daha az epoch
            validation_data=val_generator,
            callbacks=callbacks
        )
        
        # En iyi modeli değerlendir
        model.load_weights(model_save_path)
        evaluation = model.evaluate(val_generator)
        print(f"Fold {fold+1} değerlendirmesi: {dict(zip(model.metrics_names, evaluation))}")
        
        fold_metrics.append(evaluation)
        
        # Eğitim grafiklerini çiz
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f'Fold {fold+1} - Model Doğruluğu')
        plt.ylabel('Doğruluk')
        plt.xlabel('Epoch')
        plt.legend(['Eğitim', 'Doğrulama'], loc='lower right')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'Fold {fold+1} - Model Kaybı')
        plt.ylabel('Kayıp')
        plt.xlabel('Epoch')
        plt.legend(['Eğitim', 'Doğrulama'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'training_history_fold_{fold+1}.png'))
        plt.close()
    
    # K-fold performans ortalaması
    avg_metrics = np.mean(fold_metrics, axis=0)
    std_metrics = np.std(fold_metrics, axis=0)
    
    print("\n" + "="*50)
    print(f"K-fold performans özeti (ortalama ± std):")
    for i, metric_name in enumerate(model.metrics_names):
        print(f"{metric_name}: {avg_metrics[i]:.4f} ± {std_metrics[i]:.4f}")
    print("="*50)
    
    return avg_metrics

def train_with_train_test_split(data_dir, classes, output_dir):
    """
    Train-Validation-Test split yaklaşımı ile model eğitimi - ResNet50 için optimize edilmiş
    """
    # Sınırlı sayıda görüntü yükle
    print("Veri seti yükleniyor...")
    image_paths, labels = load_all_images(data_dir, classes, max_per_class=1000)
    
    # Veriyi train, validation ve test olarak böl
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=0.25, random_state=42, stratify=train_val_labels
    )
    
    print(f"Eğitim seti: {len(train_paths)} görüntü")
    print(f"Doğrulama seti: {len(val_paths)} görüntü")
    print(f"Test seti: {len(test_paths)} görüntü")
    
    # Veri jeneratörleri
    train_datagen, val_datagen = augment_data_generator()
    
    # Eğitim ve doğrulama jeneratörleri
    train_generator = create_generator_from_paths(train_paths, train_labels, train_datagen)
    val_generator = create_generator_from_paths(val_paths, val_labels, val_datagen)
    test_generator = create_generator_from_paths(test_paths, test_labels, val_datagen)
    
    # Model oluştur
    model = create_model()
    
    # Callbacks
    model_save_path = os.path.join(output_dir, 'water_meter_model_split.keras')
    callbacks = [
        ModelCheckpoint(
            model_save_path, monitor='val_accuracy', 
            save_best_only=True, mode='max', verbose=1
        ),
        EarlyStopping(
            monitor='val_loss', patience=4,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.2,
            patience=2, min_lr=1e-6, verbose=1
        )
    ]
    
    # Model eğitimi
    print("Model eğitimi başlıyor...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    # Fine-tuning
    print("Fine-tuning başlıyor...")
    model = fine_tune_model(model)
    
    history_fine = model.fit(
        train_generator,
        epochs=8,  # Fine-tuning için daha az epoch
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    # En iyi modeli yükle ve test setiyle değerlendir
    model.load_weights(model_save_path)
    test_evaluation = model.evaluate(test_generator)
    print(f"Test seti değerlendirmesi: {dict(zip(model.metrics_names, test_evaluation))}")
    
    # Eğitim grafiklerini çiz
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Doğruluğu')
    plt.ylabel('Doğruluk')
    plt.xlabel('Epoch')
    plt.legend(['Eğitim', 'Doğrulama'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Kaybı')
    plt.ylabel('Kayıp')
    plt.xlabel('Epoch')
    plt.legend(['Eğitim', 'Doğrulama'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history_split.png'))
    plt.close()
    
    return model, test_evaluation

def create_feature_dataset(data_dir, classes, max_per_class=500):
    """
    Veri kümesindeki görüntüler için özellik çıkarımı yapar - sınırlı sayıda görüntü
    """
    features_list = []
    labels = []
    processed_count = 0
    error_count = 0
    
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        class_label = 1 if class_name == 'water_meter' else 0
        
        print(f"'{class_name}' sınıfından görüntüler işleniyor...")
        
        if not os.path.exists(class_dir):
            print(f"HATA: {class_dir} dizini bulunamadı!")
            continue
            
        try:
            img_files = os.listdir(class_dir)
            # Sınıf başına görüntü sayısını sınırla
            img_files = img_files[:max_per_class]
            print(f"  {len(img_files)} adet görüntü işlenecek (maksimum {max_per_class})")
        except Exception as e:
            print(f"HATA: {class_dir} dizinindeki dosyalar listelenirken: {e}")
            continue
        
        for img_name in img_files:
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(class_dir, img_name)
            
            if processed_count % 20 == 0:  # Daha az güncellemeler
                print(f"  İşlenen görüntü sayısı: {processed_count}, hatalar: {error_count}")
                
            try:
                img = cv2.imread(img_path)
                
                if img is None:
                    error_count += 1
                    continue
                
                # Özellik çıkarımı - optimize edilmiş
                features = extract_features(img)
                features_list.append(features)
                labels.append(class_label)
                processed_count += 1
            except Exception as e:
                error_count += 1
    
    if len(features_list) == 0:
        print("UYARI: Hiçbir görüntüden özellik çıkarılamadı!")
        return pd.DataFrame()
    
    # DataFrame'e dönüştür
    print(f"Toplam {processed_count} görüntü işlendi, {error_count} hata oluştu.")
    df = pd.DataFrame(features_list)
    df['label'] = labels
    
    return df

def main():
    try:
        # Veri yolları
        data_dir = 'data_dir'  # Veri klasörü yolu
        classes = ['water_meter', 'not_water_meter']  # Sınıflar
        
        # Veri dizinini kontrol et
        if not os.path.exists(data_dir):
            print(f"HATA: {data_dir} dizini bulunamadı! Doğru veri yolunu kontrol edin.")
            return
        
        # Çıktı dizini
        output_dir = 'model_outputs'
        os.makedirs(output_dir, exist_ok=True)
        
        # Eğitim stratejisini seçme
        use_kfold = False  # K-fold cross validation için True, train-test split için False
        
        if use_kfold:
            # K-fold cross validation ile eğitim
            print("K-fold cross validation ile model eğitimi başlıyor...")
            image_paths, labels = load_all_images(data_dir, classes, max_per_class=800)
            if len(image_paths) == 0:
                print("HATA: Hiç görüntü yüklenemedi! İşlem durduruldu.")
                return
            avg_metrics = train_with_kfold(image_paths, labels, output_dir)
        else:
            # Train-test split ile eğitim - daha hızlı
            print("Train-test split ile model eğitimi başlıyor...")
            final_model, test_metrics = train_with_train_test_split(data_dir, classes, output_dir)
        
        # Özellik analizi - opsiyonel
        analyze_features = False
        if analyze_features:
            print("Özellik veri seti oluşturuluyor...")
            feature_df = create_feature_dataset(data_dir, classes, max_per_class=300)
            if not feature_df.empty:
                # Özellik dağılımlarını görselleştir
                for feature in feature_df.columns:
                    if feature != 'label':
                        plt.figure(figsize=(10, 6))
                        for label in [0, 1]:
                            subset = feature_df[feature_df['label'] == label]
                            if not subset.empty:
                                plt.hist(subset[feature], alpha=0.5, label=f"{'Su Sayacı' if label == 1 else 'Diğer'}")
                        plt.title(f"{feature} Dağılımı")
                        plt.xlabel(feature)
                        plt.ylabel('Frekans')
                        plt.legend()
                        plt.savefig(os.path.join(output_dir, f'feature_dist_{feature}.png'))
                        plt.close()
                        
                print("Özellik analizi tamamlandı.")
            
    except Exception as e:
        print(f"HATA: İşlem sırasında beklenmeyen bir hata oluştu: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
