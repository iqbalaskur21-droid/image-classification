import streamlit as st
import tensorflow as tf
import os

st.title("Klasifikasi Buah: Apple vs Mango")

# Nama folder dataset
FOLDER_DATASET = "apple"

# 1. Cek apakah folder ada
if os.path.exists(FOLDER_DATASET):
    try:
        # Load dataset
        train_ds = tf.keras.utils.image_dataset_from_directory(
            FOLDER_DATASET,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(180, 180),
            batch_size=32
        )
        
        val_ds = tf.keras.utils.image_dataset_from_directory(
            FOLDER_DATASET,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(180, 180),
            batch_size=32
        )
        
        class_names = train_ds.class_names
        st.success(f"Berhasil! Dataset dimuat.")
        st.write("Kelas yang ditemukan:", class_names)
        
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca gambar: {e}")
else:
    st.error(f"PENTING: Folder '{FOLDER_DATASET}' tidak ditemukan di GitHub.")
    st.warning("Pastikan Anda mengupload FOLDER bernama 'apple' (bukan file 'apple.txt') ke repository Anda.")
    
    # Debugging kecil untuk membantu Anda
    st.write("Isi repository saat ini:", os.listdir('.'))
