# Water Meter Detection - README

## Proje Amacı
Bu proje, bir görüntünün su sayacı olup olmadığını belirleyen bir derin öğrenme modelini içerir. Model, TensorFlow ve Keras kullanılarak eğitilmiş ve değerlendirilmiştir. Girdi olarak bir görüntü alır ve çıktıda görüntünün su sayacı olup olmadığını belirtir.

## Kurulum ve Gereksinimler
Projeyi çalıştırmadan önce aşağıdaki kütüphanelerin yüklü olduğundan emin olun:

```bash
pip install tensorflow numpy opencv-python scikit-learn
```

Ayrıca, proje dosya yapısının aşağıdaki gibi olduğundan emin olun:

```
- data_dir/
    - water_meter/
    - not_water_meter/
- train/
- val/
- test/
- my_model.keras
- threshold.txt
- results.txt
```

## Veri Seti Hazırlığı
Veri seti `data_dir/` klasörü altında iki alt kategoriye ayrılmıştır:
- `water_meter/`: Su sayacı içeren görüntüler.
- `not_water_meter/`: Su sayacı içermeyen görüntüler.

Kod, veri setini eğitim, doğrulama ve test kümelerine %80-10-10 oranında böler. Dosyalar `train/`, `val/` ve `test/` klasörlerine kopyalanır.

## Model Eğitimi
Model, 3 evrişimsel (Conv2D) ve havuzlama (MaxPooling2D) katmanına sahiptir. Sonrasında, düzleştirme (Flatten), tam bağlı (Dense) ve Dropout katmanları ile sınıflandırma yapılır.

Eğitim sırasında veri artırma (ImageDataGenerator) kullanılarak modelin genelleme yeteneği artırılır. Model, `binary_crossentropy` kaybı ve `adam` optimizasyonu ile eğitilmiştir.

Eğitim tamamlandığında model `my_model.keras` olarak kaydedilir.

## Modelin Değerlendirilmesi
Eğitilen model, test veri kümesi kullanılarak değerlendirilir. Modelin doğruluğu ve kaybı hesaplanır. Ayrıca, ROC eğrisi ve Precision-Recall eğrisi kullanılarak en iyi karar eşiği belirlenir.

Optimal eşik değeri hesaplanarak `threshold.txt` dosyasına kaydedilir.

## Tahmin Yapma
Elde edilen model kullanılarak, `test/` klasöründeki görüntüler tahmin edilir. Tahmin süreci şu adımları içerir:
1. Görüntü okunur ve ön işleme tabi tutulur (yeniden boyutlandırma, ölçekleme vb.).
2. Model ile tahmin yapılır.
3. Elde edilen olasılık değeri eşik ile karşılaştırılarak sınıflandırma yapılır.

## Sonuçların Kaydedilmesi
Her bir test görüntüsünün tahmin sonucu `results.txt` dosyasına kaydedilir. Sonuçlar aşağıdaki formatta olur:

```
image1.jpg: Image is a water meter.
image2.jpg: Image is not a water meter.
```

## Çalıştırma
Proje ana dosyasını çalıştırmak için aşağıdaki komutu kullanabilirsiniz:

```bash
python main.py
```

