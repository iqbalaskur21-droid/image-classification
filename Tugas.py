import streamlit as st
import tensorflow as tf
import os

st.title("Klasifikasi Buah")

# 1. Fungsi pencari folder yang sangat teliti
def find_dataset_folder(folder_name="apple"):
    for root, dirs, files in os.walk("."):
        if folder_name in dirs:
            return os.path.join(root, folder_name)
    return None

# 2. Mencari lokasi
data_dir = find_dataset_folder()

if data_dir:
    st.success(f"Folder ditemukan di: {data_dir}")
    
    # 3. Load dataset
    try:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(180, 180),
            batch_size=32
        )
        st.write("Dataset berhasil dimuat!")
        st.write("Kelas ditemukan:", train_ds.class_names)
    except Exception as e:
        st.error(f"Gagal memuat dataset: {e}")
else:
    # 4. Jika tetap tidak ketemu, tampilkan semua isi folder untuk perbaikan terakhir
    st.error("Folder 'apple' tidak ditemukan. Isi direktori saat ini:")
    st.write(os.listdir('.'))
    st.info("Jika folder 'apple' tidak ada di daftar di atas, berarti file/folder tersebut belum di-push ke GitHub atau berada di luar repositori yang terhubung ke Streamlit.")
