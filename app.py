import streamlit as st
import pandas as pd
import joblib # Umumnya digunakan untuk memuat model scikit-learn. Ganti dengan 'pickle' jika itu yang Anda gunakan.
# import pickle # Uncomment this if you used pickle to save your model

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Prediksi Harga Rumah KC",
    page_icon="🏠",
    layout="centered" # Atau 'wide' jika Anda ingin tata letak lebih lebar
)

# --- Judul Aplikasi ---
st.title("🏡 Prediksi Harga Rumah di King County")
st.write("Aplikasi ini memprediksi harga rumah berdasarkan fitur-fitur yang Anda masukkan.")

# --- Muat Model ---
# Pastikan file 'kc_house_data.pkl' berada di direktori yang sama dengan 'app.py'
# di dalam repositori GitHub Anda.
@st.cache_resource # Cache resource untuk memuat model hanya sekali
def load_model():
    try:
        # Ganti joblib.load dengan pickle.load jika model Anda disimpan dengan pickle
        model = joblib.load('kc_house_data.pkl')
        return model
    except FileNotFoundError:
        st.error("Error: File model 'kc_house_data.pkl' tidak ditemukan.")
        st.info("Pastikan file model Anda diunggah ke repositori GitHub yang sama dengan app.py.")
        st.stop() # Hentikan eksekusi jika model tidak ditemukan
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        st.stop()

model = load_model()
st.success("Model berhasil dimuat dan siap digunakan!")

# --- Input Pengguna (Variabel Terikat) ---
st.header("Masukkan Detail Properti untuk Prediksi")

col1, col2 = st.columns(2)

with col1:
    bedrooms = st.slider("Jumlah Kamar Tidur", min_value=0, max_value=15, value=3, step=1)
    bathrooms = st.slider("Jumlah Kamar Mandi", min_value=0.0, max_value=10.0, value=2.5, step=0.5)
    floors = st.slider("Jumlah Lantai", min_value=1.0, max_value=4.0, value=1.0, step=0.5)
    grade = st.slider("Grade (Kualitas Konstruksi & Desain)", min_value=1, max_value=13, value=7, step=1,
                      help="Grade 1 (buruk) hingga 13 (terbaik). 7 adalah rata-rata.")
    waterfront = st.checkbox("Menghadap Perairan (Waterfront)", value=False)
    # Ubah nilai boolean ke integer (0 atau 1) yang mungkin diharapkan oleh model
    waterfront_val = 1 if waterfront else 0

with col2:
    sqft_living = st.slider("Luas Ruang Hidup (sqft)", min_value=300, max_value=15000, value=2000, step=100)
    sqft_above = st.slider("Luas Di Atas Tanah (sqft)", min_value=300, max_value=15000, value=1800, step=100)
    sqft_basement = st.slider("Luas Ruang Bawah Tanah (sqft)", min_value=0, max_value=8000, value=0, step=50)
    view = st.slider("Kualitas Pemandangan", min_value=0, max_value=4, value=0, step=1,
                     help="0 (Tidak Ada), 1 (Rata-rata), 2 (Adil), 3 (Bagus), 4 (Sangat Bagus)")


# --- Tombol Prediksi ---
st.markdown("---")
if st.button("Prediksi Harga Rumah"):
    # Siapkan data input dalam format DataFrame yang diharapkan model
    # PASTIKAN NAMA KOLOM DAN URUTANNYA SAMA PERSIS DENGAN FITUR SAAT MODEL DILATIH
    input_data = pd.DataFrame([[
        bedrooms,
        bathrooms,
        floors,
        grade,
        sqft_living,
        sqft_above,
        sqft_basement,
        view,
        waterfront_val # Gunakan nilai integer
    ]], columns=['bedrooms', 'bathrooms', 'floors', 'grade', 'sqft_living',
                 'sqft_above', 'sqft_basement', 'view', 'waterfront'])

    # Tampilkan data input (opsional, untuk debugging)
    # st.write("Data Input untuk Model:")
    # st.dataframe(input_data)

    # Lakukan prediksi
    try:
        prediction = model.predict(input_data)[0].item()
        st.success(f"### Harga Rumah yang Diprediksi: ${prediction:,.2f}") # Format sebagai mata uang
        st.balloons() # Efek balon saat prediksi berhasil
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
        st.write("Pastikan input data Anda valid dan sesuai dengan ekspektasi model.")

st.markdown("---")
st.caption("Aplikasi Prediksi Harga Rumah oleh [Hanum fatikhaturrizqiana/Universitas Pekalongan]")
st.info("Catatan: Prediksi ini adalah estimasi berdasarkan model yang dilatih. Akurasi dapat bervariasi.")
