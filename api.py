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
    Su sayacı özelliklerini çıkaran geliştirilmiş fonksiyon
    """
    features = {}
    
    try:
        # Görüntü boyutunu küçült
        img = cv2.resize(img, (224, 224))
        
        # 1. Temel görüntü özellikleri
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features['brightness'] = float(np.mean(gray))
        features['contrast'] = float(np.std(gray))
        
        # 2. Mavi renk tespiti - Su sayaçları genellikle mavi tonlar içerir
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_ratio = np.sum(blue_mask > 0) / (img.shape[0] * img.shape[1])
        features['blue_ratio'] = float(blue_ratio)
        
        # 3. Dairesel şekil tespiti - Su sayaçları genellikle dairesel şekillere sahiptir
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
            param1=50, param2=30, minRadius=20, maxRadius=min(100, min(img.shape[0], img.shape[1])//2)
        )
        
        # Daire sayısı
        if circles is not None:
            features['circle_count'] = len(circles[0])
            # En büyük daireyi bul
            max_radius = 0
            for circle in circles[0]:
                if circle[2] > max_radius:
                    max_radius = circle[2]
            features['max_circle_radius'] = float(max_radius / min(img.shape[0], img.shape[1]))
        else:
            features['circle_count'] = 0
            features['max_circle_radius'] = 0.0
        
        # 4. Kenar tespiti (detay seviyesi)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
        features['edge_density'] = float(edge_density)
        
        # 5. Rakam tespiti için görüntü işleme - Su sayaçlarında rakamlar bulunur
        # Adaptif eşikleme ile rakam benzeri bölgeleri belirginleştirme
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morfolojik işlemler ile gürültüyü azalt
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Kontur bulma
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Küçük konturları filtrele - potansiyel rakam adayları
        digit_candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Rakam benzeri şekiller genellikle belirli bir boyut ve oran aralığında olur
            if 8 < w < 40 and 20 < h < 60 and 0.2 < aspect_ratio < 0.9:
                digit_candidates.append((x, y, w, h))
        
        features['digit_count'] = len(digit_candidates)
        
        # 6. Dijital gösterge bölgesi tespiti
        # Yatay olarak sıralanmış rakam benzeri şekilleri kontrol et (su sayacı göstergesi)
        if len(digit_candidates) > 2:
            digit_candidates.sort(key=lambda x: x[0])  # X koordinatına göre sırala
            horizontal_aligned = 0
            
            for i in range(len(digit_candidates)-1):
                x1, y1, w1, h1 = digit_candidates[i]
                x2, y2, w2, h2 = digit_candidates[i+1]
                
                # Yatay hizalama kontrolü - y koordinatları yaklaşık aynı, x koordinatları ardışık
                if abs(y1 - y2) < h1 * 0.5 and 0 < (x2 - (x1 + w1)) < w1 * 2:
                    horizontal_aligned += 1
            
            features['horizontal_digit_alignment'] = horizontal_aligned
        else:
            features['horizontal_digit_alignment'] = 0
        
    except Exception as e:
        print(f"Özellik çıkarma hatası: {e}")
        features = {
            'brightness': 0,
            'contrast': 0,
            'blue_ratio': 0,
            'circle_count': 0,
            'max_circle_radius': 0.0,
            'edge_density': 0,
            'digit_count': 0,
            'horizontal_digit_alignment': 0
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
        
        # GELİŞTİRİLMİŞ KARAR ALGORİTMASI
        # 1. Temel su sayacı özellikleri kontrolü
        has_circle = features['circle_count'] >= 1
        has_reasonable_edges = 0.05 < features['edge_density'] < 0.3
        has_digits = features['digit_count'] >= 3
        has_aligned_digits = features['horizontal_digit_alignment'] >= 1
        
        # 2. Su sayacı değerlendirmesi - puanlama sistemi
        score = 0
        if has_circle:
            score += 1
        if has_reasonable_edges:
            score += 1
        if has_digits:
            score += 1
        if has_aligned_digits:
            score += 2  # Yatay hizalanmış rakamlar çok önemli bir özellik
        if features['blue_ratio'] > 0.01:  # Az da olsa mavi renk varsa
            score += 1
        if features['max_circle_radius'] > 0.2:  # Büyük daire varsa (gösterge)
            score += 1
        
        # Su sayacı olması için kesin koşullar (bunlar yoksa puan ne olursa olsun sayaç değil)
        critical_features = (
            (features['circle_count'] >= 1) and                  # En az 1 daire olmalı
            (0.05 < features['edge_density'] < 0.35) and         # Belirli bir edge density aralığı
            (prediction > 0.65) and                              # Model tahmini yeterince güçlü olmalı
            (features['digit_count'] >= 3) and                   # En az 3 potansiyel rakam içermeli
            (features['horizontal_digit_alignment'] >= 1 or      # Ya yatay hizalanmış rakamlar olmalı
             features['blue_ratio'] > 0.01)                      # Ya da su sayacı mavi renk içermeli
        )
        
        # 3. Su sayacı kararı - model tahmini ve yeterli özellik puanı
        # Minimum 4 puan (7 üzerinden) VE kritik özelliklerin varlığı
        is_water_meter = score >= 4 and critical_features
        
        # Sonuçları hazırla
        result = {
            'is_water_meter': bool(is_water_meter),
            'raw_prediction': float(prediction),
            'features': features,
            'score': score,
            'analysis': {
                'has_circle': bool(has_circle),
                'has_reasonable_edges': bool(has_reasonable_edges),
                'has_digits': bool(has_digits),
                'has_aligned_digits': bool(has_aligned_digits),
                'critical_features_met': bool(critical_features)
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
        
        # GELİŞTİRİLMİŞ KARAR ALGORİTMASI
        # 1. Temel su sayacı özellikleri kontrolü
        has_circle = features['circle_count'] >= 1
        has_reasonable_edges = 0.05 < features['edge_density'] < 0.3
        has_digits = features['digit_count'] >= 3
        has_aligned_digits = features['horizontal_digit_alignment'] >= 1
        
        # 2. Su sayacı değerlendirmesi - puanlama sistemi
        score = 0
        if has_circle:
            score += 1
        if has_reasonable_edges:
            score += 1
        if has_digits:
            score += 1
        if has_aligned_digits:
            score += 2  # Yatay hizalanmış rakamlar çok önemli bir özellik
        if features['blue_ratio'] > 0.01:  # Az da olsa mavi renk varsa
            score += 1
        if features['max_circle_radius'] > 0.2:  # Büyük daire varsa (gösterge)
            score += 1
        
        # Su sayacı olması için kesin koşullar (bunlar yoksa puan ne olursa olsun sayaç değil)
        critical_features = (
            (features['circle_count'] >= 1) and                  # En az 1 daire olmalı
            (0.05 < features['edge_density'] < 0.35) and         # Belirli bir edge density aralığı
            (prediction > 0.65) and                              # Model tahmini yeterince güçlü olmalı
            (features['digit_count'] >= 3) and                   # En az 3 potansiyel rakam içermeli
            (features['horizontal_digit_alignment'] >= 1 or      # Ya yatay hizalanmış rakamlar olmalı
             features['blue_ratio'] > 0.01)                      # Ya da su sayacı mavi renk içermeli
        )
        
        # 3. Su sayacı kararı - model tahmini ve yeterli özellik puanı
        # Minimum 4 puan (7 üzerinden) VE kritik özelliklerin varlığı
        is_water_meter = score >= 4 and critical_features
        
        # Sonuçları hazırla
        result = {
            'is_water_meter': bool(is_water_meter),
            'raw_prediction': float(prediction),
            'features': features,
            'score': score,
            'analysis': {
                'has_circle': bool(has_circle),
                'has_reasonable_edges': bool(has_reasonable_edges),
                'has_digits': bool(has_digits),
                'has_aligned_digits': bool(has_aligned_digits),
                'critical_features_met': bool(critical_features)
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
