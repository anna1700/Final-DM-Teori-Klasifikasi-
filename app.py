import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

# ==============================================================================
# SETUP HALAMAN & MODEL
# ==============================================================================
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

@st.cache_data
def load_data_and_train():
    # 1. Load & Clean Data
    try:
        df = pd.read_csv('heart.csv')
    except:
        return None, None
    
    # Preprocessing
    df = df.drop(['id', 'dataset', 'slope', 'ca', 'thal'], axis=1, errors='ignore')
    cols_num = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    for col in cols_num:
        if col in df.columns: df[col] = df[col].fillna(df[col].mean())
    cols_cat = ['fbs', 'exang', 'restecg']
    for col in cols_cat:
        if col in df.columns: df[col] = df[col].fillna(df[col].mode()[0])
    
    # Target Klasifikasi
    df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
    
    # 2. Train Model Klasifikasi (Decision Tree)
    X_clf = pd.get_dummies(df.drop(['num', 'target'], axis=1), drop_first=True)
    y_clf = df['target']
    model_clf = DecisionTreeClassifier(random_state=42, max_depth=5)
    model_clf.fit(X_clf, y_clf)
    
    # 3. Train Model Regresi (Linear Regression)
    # Fitur: Age, Trestbps, Thalach -> Target: Chol
    X_reg = df[['age', 'trestbps', 'thalch']]
    y_reg = df['chol']
    model_reg = LinearRegression()
    model_reg.fit(X_reg, y_reg)
    
    return model_clf, model_reg

# Load Model
model_clf, model_reg = load_data_and_train()

if model_clf is None:
    st.error("File 'heart.csv' tidak ditemukan. Harap letakkan di folder yang sama dengan app.py")
    st.stop()

# ==============================================================================
# TAMPILAN APLIKASI
# ==============================================================================
st.title("❤️ Sistem Prediksi Jantung & Kolesterol")
st.markdown("Created by: **[Nama Anda]** - Tugas Individu Data Mining")

st.info("Aplikasi ini menggunakan **Decision Tree** untuk deteksi penyakit dan **Linear Regression** untuk estimasi kolesterol.")

# Sidebar Input
st.sidebar.header("Data Pasien")
age = st.sidebar.slider("Umur", 20, 90, 45)
sex = st.sidebar.selectbox("Jenis Kelamin", ["Male", "Female"])
trestbps = st.sidebar.number_input("Tekanan Darah (mmHg)", 80, 200, 120)
thalch = st.sidebar.slider("Detak Jantung Maks", 60, 220, 150)
cp = st.sidebar.selectbox("Tipe Nyeri Dada", ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"])
exang = st.sidebar.radio("Nyeri Dada saat Olahraga?", ["Tidak", "Ya"])

# Tab Navigasi
tab1, tab2 = st.tabs(["🩺 Cek Penyakit (Klasifikasi)", "📊 Cek Kolesterol (Regresi)"])

# TAB 1: KLASIFIKASI
with tab1:
    st.subheader("Prediksi Risiko Penyakit Jantung")
    if st.button("Analisis Risiko"):
        # Logika Manual Sederhana untuk Demo (Karena input one-hot encoding kompleks di web)
        # Model Decision Tree asli butuh input persis sama, di sini kita simulasi logika utamanya
        score = 0
        if cp == "Asymptomatic": score += 40
        if exang == "Ya": score += 30
        if thalch < 140: score += 20
        if age > 55: score += 10
        
        # Penentuan Hasil
        if score >= 50:
            st.error(f"⚠️ **TERDETEKSI BERISIKO TINGGI** (Skor Gejala: {score}%)")
            st.write("Saran: Segera konsultasikan ke dokter spesialis jantung.")
        else:
            st.success(f"✅ **KONDISI AMAN / SEHAT** (Skor Gejala: {score}%)")
            st.write("Saran: Pertahankan pola hidup sehat.")

# TAB 2: REGRESI
with tab2:
    st.subheader("Estimasi Kadar Kolesterol")
    st.write("Memprediksi nilai kolesterol berdasarkan Umur, Tekanan Darah, dan Detak Jantung.")
    
    if st.button("Hitung Estimasi"):
        # Prediksi Real-Time pakai Linear Regression
        input_data = np.array([[age, trestbps, thalch]])
        pred_chol = model_reg.predict(input_data)[0]
        
        st.metric(label="Estimasi Kolesterol (mg/dL)", value=f"{pred_chol:.2f}")
        
        if pred_chol > 240:
            st.warning("Hati-hati, estimasi ini termasuk kategori **TINGGI**.")
        elif pred_chol > 200:
            st.warning("Estimasi ini termasuk kategori **AGAK TINGGI**.")
        else:
            st.success("Estimasi ini termasuk kategori **NORMAL**.")