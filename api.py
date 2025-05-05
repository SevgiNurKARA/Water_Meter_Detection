from flask import Flask, request, jsonify
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import io
from PIL import Image

app = Flask(__name__)

# Model yükleme
MODEL_PATH = 'model_outputs/water_meter_model_split.keras'

def load_model_from_path(model_path):
    """Model dosyasını yükler"""
    try:
        if os.path.exists(model_path):
            print(f"Model yükleniyor: {model_path}")
            model = load_model(model_path)
            print("Model başarıyla yüklendi")
            return model
        else:
            print(f"HATA: Model dosyası bulunamadı: {model_path}")
            return None
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        return None

# Uygulama başladığında modeli yükle (Flask 2.0+ uyumlu)
model = None

# before_first_request yerine uygulama başlatılırken model yükleme
@app.before_request
def load_model_before_request():
    global model
    if model is None:
        model = load_model_from_path(MODEL_PATH)

def preprocess_image(image, target_size=(224, 224)):
    """Görüntüyü model için hazırlar"""
    # Boyutu ayarla
    if image.size != target_size:
        image = image.resize(target_size)
    
    # NumPy dizisine dönüştür
    img_array = img_to_array(image)
    
    # Normalizasyon
    img_array = img_array / 255.0
    
    # Batch boyutu ekle
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def extract_features(img):
    """
    Su sayacı özelliklerini çıkaran fonksiyon 
    (model.py'dan alındı ve API için uyarlandı)
    """
    features = {}
    
    try:
        # Görüntü boyutunu küçült
        img = cv2.resize(img, (224, 224))
        
        # 1. Mavi renk tespiti - Su sayaçları genellikle mavi-turkuaz tonlarında
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Daha net mavi tonu için ayarlanmış değerler (su sayacı mavisi)
        lower_blue = np.array([100, 70, 50])   # Daha doygun maviler
        upper_blue = np.array([130, 255, 255]) # Su sayacı mavisi 
        
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_ratio = np.sum(blue_mask > 0) / (img.shape[0] * img.shape[1])
        features['blue_ratio'] = float(blue_ratio)
        
        # 2. Temel görüntü özellikleri
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features['brightness'] = float(np.mean(gray))
        features['contrast'] = float(np.std(gray))
        
        # 3. Basitleştirilmiş kenar tespiti
        edges = cv2.Canny(gray, 100, 200)
        features['edge_density'] = float(np.sum(edges > 0) / (img.shape[0] * img.shape[1]))
        
        # 4. Dairesel şekil tespiti
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
            param1=50, param2=30, minRadius=20, maxRadius=min(100, min(img.shape[0], img.shape[1])//2)
        )
        
        # Daire sayısı
        circle_count = 0 if circles is None else len(circles[0])
        features['circle_count'] = circle_count
        
        # 5. Sayısal karakter tespiti (Su sayaçlarında mutlaka rakamlar vardır)
        # Görüntüyü 3x3 ızgaraya böl
        h, w = gray.shape
        cell_h, cell_w = h // 3, w // 3
        
        # Merkez bölgelerde detay olup olmadığını kontrol et
        center_edges = edges[cell_h:2*cell_h, cell_w:2*cell_w]
        center_edge_density = np.sum(center_edges > 0) / (cell_h * cell_w)
        features['center_edge_density'] = float(center_edge_density)
        
        # 6. Doku analizi 
        texture_kernel = np.ones((5,5), np.float32) / 25
        local_std = np.zeros_like(gray, dtype=np.float32)
        
        # Gri seviye varyansını hesapla (yerel doku ölçümü)
        mean_img = cv2.filter2D(gray.astype(np.float32), -1, texture_kernel)
        mean_sq_img = cv2.filter2D(np.square(gray.astype(np.float32)), -1, texture_kernel)
        local_std = np.sqrt(np.maximum(0, mean_sq_img - np.square(mean_img)))
        
        # Ortalama doku pürüzlülüğü 
        texture_roughness = np.mean(local_std)
        features['texture_roughness'] = float(texture_roughness)
        
    except Exception as e:
        print(f"Özellik çıkarma hatası: {e}")
        features = {
            'blue_ratio': 0, 
            'brightness': 0, 
            'contrast': 0, 
            'edge_density': 0,
            'circle_count': 0,
            'center_edge_density': 0,
            'texture_roughness': 0
        }
    
    return features

@app.route('/predict', methods=['POST'])
def predict():
    """Yüklenen görüntüden su sayacı tahmini yapar"""
    global model
    
    # Model yüklenmiş mi kontrol et
    if model is None:
        model = load_model_from_path(MODEL_PATH)
        if model is None:
            return jsonify({
                'error': 'Model yüklenemedi. Lütfen daha sonra tekrar deneyin.'
            }), 500
    
    # Eşik değeri parametresi (varsayılan: 0.7 - daha yüksek spesifiklik için)
    # Kullanıcı isterse farklı bir eşik değeri belirleyebilir
    confidence_threshold = request.args.get('threshold', default=0.7, type=float)
    
    # Dosya var mı kontrol et
    if 'image' not in request.files:
        return jsonify({
            'error': 'Görüntü dosyası bulunamadı.'
        }), 400
        
    file = request.files['image']
    
    try:
        # Görüntüyü oku
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # OpenCV için görüntüyü dönüştür (özellik çıkarmak için)
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Görüntü özellikleri çıkar
        features = extract_features(cv_img)
        
        # Görüntüyü ön işleme
        processed_img = preprocess_image(img)
        
        # Tahmin yap
        prediction = model.predict(processed_img, verbose=0)[0][0]
        
        # SU SAYACI TANIMLAMA KRİTERLERİ - Daha katı kurallar
        
        # 1. Model güven değeri yeterince yüksek olmalı
        model_confidence = prediction > confidence_threshold
        
        # 2. Su sayacı için gerekli kriterler
        has_circular_shape = features['circle_count'] >= 1
        has_digital_display = features['center_edge_density'] > 0.15
        has_moderate_edges = 0.1 < features['edge_density'] < 0.3
        
        # 3. Su sayacı renk şartları (ya mavi içermeli ya da gri tonlarında olmalı)
        has_specific_color = features['blue_ratio'] > 0.05
        is_grayscale_device = (features['contrast'] > 40 and features['brightness'] < 150)
        
        # 4. Doku şartları - Su sayaçları genellikle belirli seviyede doku detayına sahiptir
        has_proper_texture = 10 < features['texture_roughness'] < 30
        
        # TÜM KRİTERLER BİRLEŞTİRİLEREK SONUÇ ÜRETİLİYOR
        # Kriterlerin hepsi sağlanırsa su sayacı olarak değerlendirilir
        is_water_meter = (
            model_confidence and 
            has_circular_shape and 
            has_digital_display and 
            has_moderate_edges and
            has_proper_texture and
            (has_specific_color or is_grayscale_device)
        )
        
        # Sonuçları hazırla
        result = {
            'is_water_meter': bool(is_water_meter),
            'confidence': float(prediction),
            'applied_threshold': confidence_threshold,
            'features': features,
            'decision_factors': {
                'model_confidence': bool(model_confidence),
                'has_circular_shape': bool(has_circular_shape),
                'has_digital_display': bool(has_digital_display),
                'has_moderate_edges': bool(has_moderate_edges),
                'has_specific_color': bool(has_specific_color),
                'is_grayscale_device': bool(is_grayscale_device),
                'has_proper_texture': bool(has_proper_texture)
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': f'Tahmin hatası: {str(e)}'
        }), 500

# Postman için base64 görüntü desteği ekliyoruz
@app.route('/predict/base64', methods=['POST'])
def predict_base64():
    """Base64 kodlu görüntüden su sayacı tahmini yapar"""
    global model
    
    # Model yüklenmiş mi kontrol et
    if model is None:
        model = load_model_from_path(MODEL_PATH)
        if model is None:
            return jsonify({
                'error': 'Model yüklenemedi. Lütfen daha sonra tekrar deneyin.'
            }), 500
    
    # Eşik değeri parametresi (varsayılan: 0.7)
    confidence_threshold = request.args.get('threshold', default=0.7, type=float)
    
    # JSON verisini kontrol et
    if not request.is_json:
        return jsonify({
            'error': 'İstek JSON formatında olmalıdır.'
        }), 400
    
    data = request.get_json()
    
    if 'image' not in data:
        return jsonify({
            'error': 'Base64 kodlu görüntü bulunamadı. "image" anahtarını kullanın.'
        }), 400
        
    try:
        # Base64 görüntüyü decode et
        import base64
        img_bytes = base64.b64decode(data['image'])
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # OpenCV için görüntüyü dönüştür
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Görüntü özellikleri çıkar
        features = extract_features(cv_img)
        
        # Görüntüyü ön işleme
        processed_img = preprocess_image(img)
        
        # Tahmin yap
        prediction = model.predict(processed_img, verbose=0)[0][0]
        
        # SU SAYACI TANIMLAMA KRİTERLERİ - Daha katı kurallar
        
        # 1. Model güven değeri yeterince yüksek olmalı
        model_confidence = prediction > confidence_threshold
        
        # 2. Su sayacı için gerekli kriterler
        has_circular_shape = features['circle_count'] >= 1
        has_digital_display = features['center_edge_density'] > 0.15
        has_moderate_edges = 0.1 < features['edge_density'] < 0.3
        
        # 3. Su sayacı renk şartları (ya mavi içermeli ya da gri tonlarında olmalı)
        has_specific_color = features['blue_ratio'] > 0.05
        is_grayscale_device = (features['contrast'] > 40 and features['brightness'] < 150)
        
        # 4. Doku şartları - Su sayaçları genellikle belirli seviyede doku detayına sahiptir
        has_proper_texture = 10 < features['texture_roughness'] < 30
        
        # TÜM KRİTERLER BİRLEŞTİRİLEREK SONUÇ ÜRETİLİYOR
        is_water_meter = (
            model_confidence and 
            has_circular_shape and 
            has_digital_display and 
            has_moderate_edges and
            has_proper_texture and
            (has_specific_color or is_grayscale_device)
        )
        
        # Sonuçları hazırla
        result = {
            'is_water_meter': bool(is_water_meter),
            'confidence': float(prediction),
            'applied_threshold': confidence_threshold,
            'features': features,
            'decision_factors': {
                'model_confidence': bool(model_confidence),
                'has_circular_shape': bool(has_circular_shape),
                'has_digital_display': bool(has_digital_display),
                'has_moderate_edges': bool(has_moderate_edges),
                'has_specific_color': bool(has_specific_color),
                'is_grayscale_device': bool(is_grayscale_device),
                'has_proper_texture': bool(has_proper_texture)
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': f'Tahmin hatası: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """API sağlık kontrolü"""
    model_loaded = model is not None
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded
    })

if __name__ == '__main__':
    # Debug modunda çalıştır (üretimde False yapılmalı)
    app.run(debug=True, host='0.0.0.0', port=5000) 
