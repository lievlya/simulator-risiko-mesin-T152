import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. LOAD MODEL & SCALER (Prinsip Inference)
# ==========================================
# Memanggil file fisik yang sudah disimpan sebelumnya (efisiensi energi & etika)
model = joblib.load('model_risiko_v1.joblib')
scaler = joblib.load('scaler_risiko_v1.joblib')

# ==========================================
# 2. ANTARMUKA DASHBOARD (Streamlit UI)
# ==========================================
st.title("Simulator Risiko Kegagalan Sistem 🚀")
st.write("Aplikasi ini memprediksi skor risiko mesin berdasarkan input sensor secara real-time.")

st.sidebar.header("Input Data Sensor")

# Input dari pengguna
suhu_input = st.sidebar.number_input("Suhu Mesin (°C)", min_value=0.0, max_value=250.0, value=85.0)
getaran_input = st.sidebar.number_input("Getaran Mesin (mm/s)", min_value=0.0, max_value=100.0, value=7.0)

# ==========================================
# 3. MONITORING DRIFT (Kesehatan Model)
# ==========================================
# Validasi apakah input pengguna keluar dari jangkauan data latihan awal
if suhu_input > 120 or suhu_input < 10:
    st.warning("⚠️ Input di luar jangkauan data latihan. Hasil simulasi mungkin tidak akurat!")

# ==========================================
# 4. PROSES PREDIKSI (Inference Flow)
# ==========================================
if st.button("Jalankan Simulasi Risiko"):
    # Format data sesuai kebutuhan model [[Suhu, Getaran]]
    data_baru = np.array([[suhu_input, getaran_input]])
    
    # Transformasi data menggunakan scaler bawaan (Konsistensi Scaling)
    data_baru_scaled = scaler.transform(data_baru)
    
    # Prediksi menggunakan model
    skor_risiko = model.predict(data_baru_scaled)
    
    # Tampilkan Hasil
    st.subheader("Hasil Analisis")
    st.metric(label="Skor Risiko Kegagalan", value=f"{skor_risiko[0]:.2f}")
    
    # Batas aman visual sederhana
    if skor_risiko[0] > 50:
        st.error("Status: RISIKO TINGGI! Perlu tindakan pemeliharaan segera.")
    else:
        st.success("Status: Sistem Aman / Stabil.")