# Water Meter Detection API

## Proje Amacı
Bu proje, bir görüntünün su sayacı olup olmadığını belirleyen bir derin öğrenme modelini içerir. Model, ResNet50 mimarisi temel alınarak TensorFlow ve Keras kullanılarak eğitilmiş ve değerlendirilmiştir. Girdi olarak bir görüntü alır ve çıktıda görüntünün su sayacı olup olmadığını belirtir.

## Kurulum ve Gereksinimler

API'yi çalıştırmak için aşağıdaki adımları izleyin:

1. Gerekli paketleri yükleyin:
```bash
pip install tensorflow numpy opencv-python scikit-learn flask pillow
```

2. Model dosyasının `model_outputs/water_meter_model_split.keras` konumunda olduğundan emin olun.

3. API'yi başlatın:
```bash
python api.py
```

API varsayılan olarak 5000 portunda çalışacaktır.

## Proje Dosya Yapısı

```
- data_dir/
    - water_meter/       # Su sayacı içeren görüntüler
    - not_water_meter/   # Su sayacı içermeyen görüntüler
- train/                 # Eğitim veri seti
- val/                   # Doğrulama veri seti
- test/                  # Test veri seti
- model_outputs/
    - water_meter_model_split.keras  # Eğitilmiş model
- api.py                 # API kodu
- requirements.txt       # Gerekli kütüphaneler
```

## Veri Seti Hazırlığı
Veri seti `data_dir/` klasörü altında iki alt kategoriye ayrılmıştır:
- `water_meter/`: Su sayacı içeren görüntüler.
- `not_water_meter/`: Su sayacı içermeyen görüntüler.

Kod, veri setini eğitim, doğrulama ve test kümelerine %80-10-10 oranında böler. Dosyalar `train/`, `val/` ve `test/` klasörlerine kopyalanır.

## Model Eğitimi
Model, önceden eğitilmiş ResNet50 mimarisi üzerine inşa edilmiştir. Transfer öğrenme yaklaşımı kullanılarak, su sayacı tespiti için özelleştirilmiştir. Model aşağıdaki katmanları içerir:

1. ResNet50 temel modeli (ImageNet ağırlıkları ile)
2. Global Average Pooling katmanı
3. Tam bağlı (Dense) katmanlar
4. Dropout katmanları (aşırı öğrenmeyi önlemek için)
5. Sigmoid aktivasyon fonksiyonu ile çıkış katmanı

Eğitim sırasında veri artırma (ImageDataGenerator) kullanılarak modelin genelleme yeteneği artırılır:
- Rastgele döndürme
- Yatay ve dikey çevirme
- Yakınlaştırma
- Parlaklık değişimleri

Model, `binary_crossentropy` kaybı ve `adam` optimizasyonu ile eğitilmiştir.

## Modelin Değerlendirilmesi
Eğitilen model, test veri kümesi kullanılarak değerlendirilir. Modelin doğruluğu, hassasiyeti (precision), duyarlılığı (recall) ve F1 skoru hesaplanır. Ayrıca, ROC eğrisi ve Precision-Recall eğrisi kullanılarak en iyi karar eşiği belirlenir.

Optimal eşik değeri hesaplanarak `threshold.txt` dosyasına kaydedilir.

## API Kullanımı

### Sağlık Kontrolü

API'nin çalışıp çalışmadığını kontrol etmek için:

```bash
curl http://localhost:5000/health
```

Beklenen cevap:
```json
{
  "is_healthy": true,
  "model_path": "model_outputs/water_meter_model_split.keras",
  "status": "API çalışıyor, model yüklü"
}
```

### Görüntü Dosyası ile Tahmin

```
POST /predict
```

Multipart form veri olarak bir görüntü dosyası göndererek tahmin yapar.

**Parametreler:**
- `image`: Görüntü dosyası (JPG, PNG)

**Örnek İstek (curl):**
```bash
curl -X POST -F "image=@/path/to/meter_image.jpg" http://localhost:5000/predict
```

**Örnek Yanıt:**
```json
{
  "confidence": 0.9876,
  "features": {
    "blue_ratio": 0.123,
    "brightness": 145.67,
    "contrast": 56.78,
    "edge_density": 0.234
  },
  "is_water_meter": true
}
```

### Base64 Kodlu Görüntü ile Tahmin

```
POST /predict/base64
```

JSON içinde Base64 kodlu bir görüntü göndererek tahmin yapar.

**Parametreler:**
- JSON gövdesi içinde `image`: Base64 kodlu görüntü

**Örnek İstek (curl):**
```bash
curl -X POST -H "Content-Type: application/json" -d '{"image":"base64_encoded_image_data"}' http://localhost:5000/predict/base64
```

**Örnek Yanıt:**
```json
{
  "confidence": 0.9876,
  "features": {
    "blue_ratio": 0.123,
    "brightness": 145.67,
    "contrast": 56.78,
    "edge_density": 0.234
  },
  "is_water_meter": true
}
```

## Test Örneği

API'yi test etmek için Python ile örnek bir istek:

```python
import requests

# Dosya ile test
with open("test_image.jpg", "rb") as f:
    response = requests.post("http://localhost:5000/predict", files={"image": f})
    print(response.json())

# Base64 ile test
import base64
with open("test_image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')
    response = requests.post(
        "http://localhost:5000/predict/base64",
        json={"image": image_data}
    )
    print(response.json())
```

## Docker ile Kullanım

Docker kullanarak API'yi çalıştırmak için:

1. Docker imajı oluşturun:
```bash
docker build -t water-meter-api .
```

2. Docker konteynerını çalıştırın:
```bash
docker run -p 5000:5000 water-meter-api
```

## Notlar

- API, görüntüleri 224x224 boyutuna yeniden boyutlandırır ve normalize eder.
- Tahmin sonucunda belirlenen eşik değerinden büyük sonuçlar "su sayacı var" olarak yorumlanır.
- `features` alanı, görüntüden çıkarılan öznitelikleri gösterir ve analiz için ek bilgi sağlar.

## API Endpointleri

### 1. Sağlık Kontrolü

```
GET /health
```

API'nin çalışıp çalışmadığını ve modelin yüklenip yüklenmediğini kontrol eder.

**Örnek Yanıt:**
```json
{
  "is_healthy": true,
  "model_path": "model_outputs/water_meter_model_split.keras",
  "status": "API çalışıyor, model yüklü"
}
```

### 2. Görüntü Dosyası ile Tahmin

```
POST /predict
```

Multipart form veri olarak bir görüntü dosyası göndererek tahmin yapar.

**Parametreler:**
- `image`: Görüntü dosyası (JPG, PNG)

**Örnek İstek (curl):**
```bash
curl -X POST -F "image=@/path/to/meter_image.jpg" http://localhost:5000/predict
```

**Örnek Yanıt:**
```json
{
  "confidence": 0.9876,
  "features": {
    "blue_ratio": 0.123,
    "brightness": 145.67,
    "contrast": 56.78,
    "edge_density": 0.234
  },
  "is_water_meter": true
}
```

### 3. Base64 Kodlu Görüntü ile Tahmin

```
POST /predict/base64
```

JSON içinde Base64 kodlu bir görüntü göndererek tahmin yapar.

**Parametreler:**
- JSON gövdesi içinde `image`: Base64 kodlu görüntü

**Örnek İstek (curl):**
```bash
curl -X POST -H "Content-Type: application/json" -d '{"image":"base64_encoded_image_data"}' http://localhost:5000/predict/base64
```

**Örnek Yanıt:**
```json
{
  "confidence": 0.9876,
  "features": {
    "blue_ratio": 0.123,
    "brightness": 145.67,
    "contrast": 56.78,
    "edge_density": 0.234
  },
  "is_water_meter": true
}
```

## Test

API'yi test etmek için Python ile örnek bir istek:

```python
import requests

# Dosya ile test
with open("test_image.jpg", "rb") as f:
    response = requests.post("http://localhost:5000/predict", files={"image": f})
    print(response.json())

# Base64 ile test
import base64
with open("test_image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')
    response = requests.post(
        "http://localhost:5000/predict/base64",
        json={"image": image_data}
    )
    print(response.json())
``` 
