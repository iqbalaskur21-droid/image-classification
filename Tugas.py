import streamlit as st
from PIL import Image

st.title("Klasifikasi Buah: Apple vs Mango")

# Menampilkan gambar yang ada
st.image("apple1.jpg", caption="Ini adalah Apple")
st.image("mango2.jpg", caption="Ini adalah Mango")

# Contoh logika sederhana untuk klasifikasi
pilihan = st.selectbox("Pilih buah untuk diidentifikasi:", ["Apple", "Mango"])

if st.button("Deteksi"):
    if pilihan == "Apple":
        st.success("Hasil: Ini adalah buah Apple")
    else:
        st.success("Hasil: Ini adalah buah Mango")
