import streamlit as st  # Library untuk membuat UI web interaktif
from ultralytics import YOLO  # Library YOLO untuk klasifikasi objek
import numpy as np  # Untuk manipulasi array gambar
from PIL import Image  # Untuk membuka dan memproses gambar
import os  # Untuk operasi file dan folder

# =========================
# Path ke file model YOLO
# =========================
MODEL_PATH = "models/best_model.pt"

# =========================
# Cek keberadaan model
# =========================
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model tidak ditemukan di {MODEL_PATH}")  # Tampilkan error jika file model tidak ada
else:
    model = YOLO(MODEL_PATH)  # Load model YOLO
    class_names = model.names  # Ambil daftar nama kelas dari model

    # =========================
    # Streamlit UI
    # =========================
    st.title("🍽️ Klasifikasi Peralatan Makan (YOLO11)")  # Judul aplikasi

    # Komponen upload file gambar
    uploaded_file = st.file_uploader("Upload Gambar", type=["jpg","png","jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)  # Buka gambar yang diupload
        st.image(image, caption="Gambar yang di-upload", use_column_width=True)  # Tampilkan gambar

        image_np = np.array(image)  # Konversi gambar ke array numpy untuk diproses YOLO

        results = model(image_np)  # Lakukan prediksi menggunakan model
        probs = results[0].probs  # Ambil probabilitas hasil prediksi

        if probs is not None:
            # Ambil 3 prediksi teratas
            top_indices = probs.top5[:3]  
            top_scores = probs.data[top_indices].cpu().numpy()  # Konversi ke numpy untuk ditampilkan
            predictions = [f"{class_names[int(idx)]} : {score:.4f}"
                           for idx, score in zip(top_indices, top_scores)]  # Format prediksi

            st.subheader("Top-3 Prediksi")  # Subjudul hasil prediksi
            for pred in predictions:
                st.write(pred)  # Tampilkan prediksi
        else:
            st.write("Tidak ada prediksi yang valid")  # Jika prediksi kosong
