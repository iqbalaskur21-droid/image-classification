import streamlit as st
import tensorflow as tf
import os

st.title("Klasifikasi Buah: Apple vs Mango")

# Pastikan folder 'apple' berada di level yang sama dengan file Tugas.py
data_dir = "apple" 

if os.path.exists(data_dir):
    try:
        # Load dataset dengan konfigurasi standar
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(180, 180),
            batch_size=32
        )
        
        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(180, 180),
            batch_size=32
        )
        
        class_names = train_ds.class_names
        st.success(f"Dataset berhasil dimuat! Kelas yang ditemukan: {class_names}")
        
        # Tampilkan informasi singkat
        st.write("Jumlah kelas:", len(class_names))
        
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat gambar: {e}")
else:
    st.error(f"Folder '{data_dir}' tidak ditemukan di GitHub. Pastikan folder tersebut ada di repositori Anda.")
