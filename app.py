import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from keras import layers, models, Input
from tensorflow.keras.preprocessing import image

# ==========================================
# 1. SETUP HALAMAN
# ==========================================
st.set_page_config(page_title="Ekspresi Wajah Real-Time", page_icon="😀", layout="centered")

st.title("😀 Deteksi Ekpresi Wajah")
st.write("Model CNN + CBAM akan mendeteksi ekspresi secara otomatis.")

# ==========================================
# 2. DEFINISI ARSITEKTUR MODEL (Wajib)
# ==========================================
def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]
    shared_layer_one = layers.Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    shared_layer_two = layers.Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    
    avg_pool = layers.GlobalAveragePooling2D()(input_feature)    
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    
    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    
    cbam_feature = layers.Add()([avg_pool, max_pool])
    cbam_feature = layers.Activation('sigmoid')(cbam_feature)
    return layers.Multiply()([input_feature, cbam_feature])

def spatial_attention(input_feature):
    avg_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(input_feature)
    max_pool = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(input_feature)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    cbam_feature = layers.Conv2D(filters=1, kernel_size=7, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)
    return layers.Multiply()([input_feature, cbam_feature])

def cbam_block(cbam_feature, ratio=8):
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

def build_model():
    input_layer = Input(shape=(150, 150, 1))

    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
    x = layers.BatchNormalization()(x)
    x = cbam_block(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = cbam_block(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = cbam_block(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = cbam_block(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x) 
    output_layer = layers.Dense(7, activation='softmax')(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    return model

# ==========================================
# 3. LOAD MODEL & CONFIG
# ==========================================
@st.cache_resource
def load_emotion_model():
    model = build_model()
    try:
        model.load_weights('fer_model.h5') 
        print("✅ Model Loaded!")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
    return model

model = load_emotion_model()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Palet Warna (BGR)
emotion_colors = {
    'Angry': (0, 0, 255),     # Merah
    'Disgust': (0, 128, 0),   # Hijau Gelap
    'Fear': (128, 0, 128),    # Ungu
    'Happy': (0, 255, 255),   # Kuning
    'Neutral': (255, 255, 255), # Putih
    'Sad': (255, 0, 0),       # Biru
    'Surprise': (0, 165, 255) # Oranye
}

# ==========================================
# 4. LOOP KAMERA OTOMATIS
# ==========================================
FRAME_WINDOW = st.image([]) # Placeholder gambar
camera = cv2.VideoCapture(0)

# Loop akan berjalan terus selama aplikasi dibuka
while True:
    ret, frame = camera.read()
    if not ret:
        st.warning("Kamera tidak terdeteksi.")
        break
    
    frame = cv2.flip(frame, 1) # Mirror effect
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        try:
            roi = cv2.resize(face_roi, (150, 150))
            roi = roi.astype("float") / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)

            # Prediksi
            preds = model.predict(roi, verbose=0)[0]
            label_idx = np.argmax(preds)
            label = class_labels[label_idx]
            confidence = preds[label_idx] * 100

            # Visualisasi Warna
            color = emotion_colors.get(label, (0, 255, 0))

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(frame, (x, y-35), (x+w, y), color, -1)
            cv2.putText(frame, f"{label} ({confidence:.0f}%)", (x + 5, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        
        except Exception:
            pass

    # Update tampilan di Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame_rgb)
