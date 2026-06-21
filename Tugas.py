from PIL import Image
import streamlit as st

# --- KODE LAMA (JANGAN DIGUNAKAN) ---
# img = Image.open("fruits/apple1.jpg") 

# --- KODE BARU (GUNAKAN INI) ---
# Mengambil file langsung dari lokasi yang sama dengan script
try:
    apple_img = Image.open("apple1.jpg")
    mango_img = Image.open("mango2.jpg")
    
    st.image(apple_img, caption="Apple")
    st.image(mango_img, caption="Mango")
except FileNotFoundError:
    st.error("File gambar tidak ditemukan. Pastikan nama file sesuai: apple1.jpg dan mango2.jpg")
