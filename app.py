import streamlit as st
import pandas as pd
import joblib # Atau pickle, tergantung bagaimana Anda menyimpan model

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Prediksi Harga Rumah KC",
    page_icon="🏠",
    layout="centered"
)

# --- Judul Aplikasi ---
st.title("🏡 Prediksi Harga Rumah di King County")
st.write("Aplikasi ini memprediksi harga rumah berdasarkan beberapa fitur kunci.")

# --- Muat Model ---
# Pastikan file 'kc_house_data.pkl' berada di direktori yang sama dengan 'app.py'
# di dalam repositori GitHub Anda.
try:
    model = joblib.load('kc_house_data.pkl')
    st.success("Model berhasil dimuat!")
except FileNotFoundError:
    st.error("Error: File model 'kc_house_data.pkl' tidak ditemukan. Pastikan file ada di repositori GitHub Anda.")
    st.stop() # Hentikan eksekusi jika model tidak ditemukan
except Exception as e:
    st.error(f"Error saat memuat model: {e}")
    st.stop()

# --- Input Pengguna (Contoh Fitur) ---
st.header("Masukkan Fitur Rumah")

# Contoh fitur: square footage, number of bedrooms, number of bathrooms
# Sesuaikan ini dengan fitur sebenarnya yang digunakan model Anda!
sqft_living = st.slider("Luas Hidup (sqft)", min_value=500, max_value=8000, value=2000, step=50)
bedrooms = st.slider("Jumlah Kamar Tidur", min_value=1, max_value=10, value=3, step=1)
bathrooms = st.slider("Jumlah Kamar Mandi", min_value=0.5, max_value=8.0, value=2.5, step=0.5)
# Anda bisa menambahkan lebih banyak input sesuai fitur model Anda (e.g., year_built, grade, view, dll.)
# Contoh:
# year_built = st.number_input("Tahun Dibangun", min_value=1900, max_value=2025, value=2000)


# --- Tombol Prediksi ---
if st.button("Prediksi Harga"):
    # Buat DataFrame dari input pengguna
    # Pastikan nama kolom sesuai dengan yang diharapkan oleh model Anda!
    input_data = pd.DataFrame([[sqft_living, bedrooms, bathrooms]],
                              columns=['sqft_living', 'bedrooms', 'bathrooms']) # Ganti dengan nama fitur model Anda

    # Lakukan prediksi
    try:
        prediction = model.predict(input_data)[0]
        st.subheader("Hasil Prediksi Harga Rumah:")
        st.success(f"Harga Rumah yang Diprediksi: ${prediction:,.2f}") # Format sebagai mata uang
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
        st.write("Pastikan input data dan format fitur sesuai dengan yang diharapkan model Anda.")

st.markdown("---")
st.markdown("Dibuat oleh Data Scientist Anda")
