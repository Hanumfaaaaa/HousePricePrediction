import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("Prediksi Harga Rumah 🏠")

# Load data
df = pd.read_csv("kc_house_data.csv")

# Pastikan kolom-kolom ada
fitur = ['bedrooms', 'bathrooms', 'floors', 'grade', 'sqft_living',
         'sqft_above', 'sqft_basement', 'view', 'waterfront']
df = df[fitur + ['price']].dropna()

# Model
model = LinearRegression()
X = df[fitur]
y = df['price']
model.fit(X, y)

# Input pengguna
st.sidebar.header("Masukkan data rumah:")
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 8, 2)
floors = st.sidebar.slider("Floors", 1, 3, 1)
grade = st.sidebar.slider("Grade", 1, 13, 7)
sqft_living = st.sidebar.slider("Sqft Living", 500, 10000, 2000)
sqft_above = st.sidebar.slider("Sqft Above", 500, 10000, 1500)
sqft_basement = st.sidebar.slider("Sqft Basement", 0, 5000, 500)
view = st.sidebar.slider("View", 0, 4, 0)
waterfront = st.sidebar.selectbox("Waterfront (1=Ya, 0=Tidak)", [0, 1])

# Prediksi
input_data = [[bedrooms, bathrooms, floors, grade, sqft_living,
               sqft_above, sqft_basement, view, waterfront]]

prediksi = model.predict(input_data)[0]
st.success(f"💰 Estimasi harga rumah: ${prediksi:,.0f}")
