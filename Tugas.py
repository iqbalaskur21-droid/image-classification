import streamlit as st
import tensorflow as tf
import os

st.title("Klasifikasi Buah: Apple vs Mango")

# Fungsi mencari folder 'apple' secara otomatis di dalam repository
def find_data_dir(folder_name="apple"):
    for root, dirs, files in os.walk("."):
        if folder_name in dirs:
            return os.path.join(root, folder_name)
    return None

data_dir = find_data_dir()

if data_dir:
    try:
        # Load dataset
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
        
        st.success(f"Dataset berhasil dimuat dari: {data_dir}")
        st.write("Kelas yang ditemukan:", train_ds.class_names)
        
    except Exception as e:
        st.error(f"Error saat load dataset: {e}")
else:
    st.error("Folder 'apple' tidak ditemukan. Pastikan folder tersebut ada di GitHub dan berisi sub-folder kelas (apple & mango).")
