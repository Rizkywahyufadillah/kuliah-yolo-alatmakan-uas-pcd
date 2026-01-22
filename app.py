# =========================
# app.py - Streamlit YOLO11 Klasifikasi Alat Makan
# =========================

import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os

# =========================
# Load Model YOLO
# =========================
# Pastikan file model sudah ada di folder 'models/best_model.pt'
MODEL_PATH = "models/best_model.pt"

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model tidak ditemukan di {MODEL_PATH}. Silakan letakkan file best_model.pt di folder 'models'.")
else:
    model = YOLO(MODEL_PATH)
    class_names = model.names

    # =========================
    # Streamlit UI
    # =========================
    st.title("🍽️ Klasifikasi Peralatan Makan (YOLO11)")

    uploaded_file = st.file_uploader("Upload Gambar Peralatan Makan", type=["jpg","png","jpeg"])

    if uploaded_file:
        # Buka gambar
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang di-upload", use_column_width=True)

        # Convert ke numpy array
        image_np = np.array(image)

        # Prediksi
        results = model(image_np)
        probs = results[0].probs

        if probs is not None:
            top_indices = probs.top5[:3]
            top_scores = probs.data[top_indices].cpu().numpy()
            predictions = [f"{class_names[int(idx)]} : {score:.4f}"
                           for idx, score in zip(top_indices, top_scores)]

            st.subheader("Top-3 Prediksi")
            for pred in predictions:
                st.write(pred)
        else:
            st.write("Tidak ada prediksi yang valid")
