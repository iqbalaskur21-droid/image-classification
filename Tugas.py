import streamlit as st
import tensorflow as tf
import os
import pathlib

# Fungsi untuk mencari folder dataset secara otomatis di lokasi proyek
def get_dataset_path(target_folder="apple"):
    # Mencari folder 'apple' di seluruh sub-direktori
    for root, dirs, files in os.walk("."):
        if target_folder in dirs:
            return os.path.join(root, target_folder)
    return None

st.title("Aplikasi Klasifikasi Buah")

# Mencari lokasi folder dataset
data_dir = get_dataset_path()

if data_dir:
    st.write(f"Dataset ditemukan di: {data_dir}")
    
    batch_size = 32
    img_height = 180
    img_width = 180

    # Load dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    st.write("Class names:", class_names)

    # Optimasi performa
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Arsitektur Model CNN
    num_classes = len(class_names)
    model = tf.keras.Sequential([
      tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
      tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Training (Hanya jika tombol ditekan untuk menghindari crash saat load awal)
    if st.button("Mulai Training Model"):
        with st.spinner("Sedang training..."):
            epochs = 5 # Disesuaikan agar tidak terlalu lama di cloud
            history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
            st.success("Training selesai!")
            st.write(history.history)

else:
    st.error("Folder 'apple' tidak ditemukan di repositori GitHub Anda. Mohon pastikan folder sudah di-upload!")
