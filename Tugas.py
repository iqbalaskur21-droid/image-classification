import streamlit as st
import tensorflow as tf
import os

st.title("Klasifikasi Buah: Apple vs Mango")

# Nama folder utama yang ada di GitHub Anda
DATA_DIR = "fruits" 

# Cek apakah folder 'fruits' ada
if os.path.exists(DATA_DIR):
    try:
        # Memuat dataset dari folder 'fruits'
        # Struktur: fruits/apple/... dan fruits/mango/...
        train_ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(180, 180),
            batch_size=32
        )
        
        val_ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(180, 180),
            batch_size=32
        )
        
        class_names = train_ds.class_names
        st.success(f"Dataset berhasil dimuat dari folder '{DATA_DIR}'!")
        st.write("Kelas yang ditemukan:", class_names)
        
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat gambar: {e}")
else:
    st.error(f"Folder '{DATA_DIR}' tidak ditemukan.")
    st.info("Pastikan Anda sudah mengupload FOLDER bernama 'fruits' ke GitHub.")
    st.write("Isi folder saat ini:", os.listdir('.'))
