# 😀 Facial Emotion Recognition (CNN + CBAM)

Aplikasi deteksi ekspresi wajah real-time berbasis web menggunakan **Streamlit**. Model ini dibangun menggunakan arsitektur **Custom CNN** yang diperkuat dengan **CBAM (Convolutional Block Attention Module)** agar lebih fokus pada fitur wajah penting (seperti mata dan mulut).

## ✨ Fitur Utama

* **Real-time Detection:** Mendeteksi wajah dan ekspresi langsung dari Webcam.
* **CBAM Attention:** Meningkatkan akurasi dengan mekanisme atensi spasial dan channel.
* **7 Kelas Ekspresi:** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.
* **Interactive Dashboard:** Visualisasi persentase ekspresi dengan grafik batang dinamis.

## 🛠️ Tech Stack

* **Python 3.x**
* **TensorFlow / Keras** (Model Training & Inference)
* **OpenCV** (Face Detection & Image Processing)
* **Streamlit** (Web Interface)
* **Numpy**

## 🚀 Cara Menjalankan

1. **Clone repo ini** atau letakkan semua file dalam satu folder.
2. **Pastikan struktur folder:**
```text
📁 facial_expression_recognation_cban_x_cnn/
├── 📄 app.py
├── 📄 facial_expression_recognation_cban_x_cnn.ipynb
├── 📦 fer_model.h5
├── 📄 .gitignore
├── 📄 README.md
└── 📄 requirements.txt

```


3. **Install Library:**
Buka terminal dan jalankan:
```bash
pip install tensorflow streamlit opencv-python numpy

```

atau 

```bash
pip install -r requirements.txt

```


4. **Jalankan Aplikasi:**
```bash
streamlit run app.py

```


5. **Selesai!** Browser akan otomatis terbuka di `http://localhost:8501`.

## 🧠 Arsitektur Model

Model menggunakan 4 blok Konvolusi, di mana setiap blok diikuti oleh **CBAM Block** sebelum dilakukan *Max Pooling*. Ini memaksa model untuk belajar "di mana" harus melihat (Spatial) dan "apa" yang penting (Channel).

Dilatih menggunakan Google Colab menggunakan GPU T4 dengan Ekstra RAM.