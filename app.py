import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
import network
import cv2

# ==========================================
# 1. INITIALIZATION & LAYOUT CONFIGURATION
# ==========================================

# Memuat model Neural Network yang sudah dilatih dari file pkl
# MODEL EMNIST 36 CLASS
net = network.Network.load("trained_network_emnist.pkl")

# MAPPING LABEL EMNIST
CLASS_NAMES = [
    '0','1','2','3','4',
    '5','6','7','8','9',
    'A','B','C','D','E',
    'F','G','H','I','J',
    'K','L','M','N','O',
    'P','Q','R','S','T',
    'U','V','W','X','Y','Z'
]

# Mengatur konfigurasi dasar halaman web Streamlit
st.set_page_config(page_title="Multi-Digit Classifier", layout="centered")
st.title("Digit Classifier Neural Network (Multi-Digit Support)")

# Menentukan ukuran tampilan kanvas (280x280 piksel)

CANVAS_SIZE = 280

# Membagi layout web menjadi 2 kolom (Kiri untuk Input, Kanan untuk Visualisasi AI)
left_col, right_col = st.columns([1, 1])

# Membuat komponen kanvas gambar di kolom kiri
with left_col:
    st.markdown("#### Input Canvas")
    canvas_result = st_canvas(
        fill_color="#FFFFFF",         # Warna kuas bagian dalam (jika membuat bentuk geometris)
        stroke_width=16,              # Ketebalan garis kuas tulisan tangan
        stroke_color="#FFFFFF",        # Warna kuas putih (karena MNIST menggunakan background hitam garis putih)
        background_color="#000000",   # Warna dasar kanvas hitam
        height=CANVAS_SIZE,
        width=CANVAS_SIZE,
        drawing_mode="freedraw",       # Mode menggambar bebas menggunakan mouse/touchpad
        key="canvas",
        update_streamlit=True,        # Memperbarui data secara real-time saat user menggambar
    )

# ==========================================
# 2. IMAGE PREPROCESSING (MNIST STANDARD)
# ==========================================
def process_single_digit(cropped_img):
    """
    Fungsi untuk mengubah satu potongan gambar angka acak menjadi format standar MNIST (28x28 piksel,
    angka berada tepat di tengah dengan margin proporsional, nilai piksel dinormalisasi 0-1).
    """

    w, h = cropped_img.shape[1], cropped_img.shape[0]
    # Menghitung skala agar sisi terpanjang angka menjadi 20 piksel (meniru proporsi standar MNIST)
    scale = 20.0 / max(w, h)
    resized = cv2.resize(cropped_img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    # Membuat kanvas kosong baru berukuran 28x28 piksel dengan warna dasar hitam (nilai 0)
    result = np.zeros((28, 28), dtype=np.float32)
    # Menghitung koordinat agar potongan angka berada tepat di tengah-tengah kanvas 28x28
    x_offset = (28 - resized.shape[1]) // 2
    y_offset = (28 - resized.shape[0]) // 2
    result[y_offset:y_offset+resized.shape[0], x_offset:x_offset+resized.shape[1]] = resized
    # Normalisasi nilai piksel dari rentang (0-255) menjadi (0.0 - 1.0) agar stabil saat diproses AI
    return result / 255.0

# ==========================================
# 3. PREDICTION & SEGMENTATION LOGIC
# ==========================================

# Aksi yang dipicu ketika tombol "Check digit" diklik
if st.button("Check digit"):
    # Memastikan data gambar dari kanvas tersedia
    if canvas_result.image_data is not None:
        # Mengubah data kanvas menjadi array matriks bertipe 8-bit unsigned integer (0-255)
        rgba = canvas_result.image_data.astype(np.uint8)

        # Mengubah gambar RGBA (4 channel warna) menjadi Grayscale/Hitam-Putih (1 channel warna)
        gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
        # Thresholding: Mengubah piksel abu-abu samar menjadi putih tegas jika nilainya di atas 50
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        # Kontur: Mencari area/objek angka terpisah yang ada di dalam kanvas
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Jika tidak ditemukan coretan objek sama sekali
        if len(contours) == 0:
            st.warning("Please draw a digit first.")
        else:
            # Membuat kotak pembatas (Bounding Box) untuk setiap objek kontur yang ditemukan
            bouding_boxes = [cv2.boundingRect(c) for c in contours]

            # Urutkan kotak pembatas dari KIRI ke KANAN berdasarkan koordinat X
            # (Penting agar angka multi-digit seperti '10' tidak dibaca terbalik menjadi '01')
            bouding_boxes = sorted(bouding_boxes, key=lambda b: b[0])

            # Inisialisasi wadah penampung hasil proses per-digit
            predictions = []    # Menyimpan teks hasil tebakan (misal: ['1', '0'])
            display_images = [] # Menyimpan gambar hasil normalisasi 28x28 untuk divisualisasikan
            outputs = []        # Menyimpan matriks probabilitas output dari AI (untuk diagram batang)
            # Melakukan perulangan (looping) untuk memproses tiap angka satu per satu
            for bbox in bouding_boxes:
                x, y, w, h = bbox
                # Proteksi: Abaikan jika objek terlalu kecil (mengeliminasi noda/titik tidak sengaja)
                if w < 5 or h < 5:
                    continue
                # Potong gambar secara pas hanya di area angka tersebut saja
                digit_crop = thresh[y:y+h, x:x+w]
                # Proses potongan angka tersebut agar sesuai format MNIST (28x28)
                processed_digit = process_single_digit(digit_crop)
                display_images.append(processed_digit)
                # Flattening: Mengubah matriks 28x28 menjadi vektor baris tunggal 784x1 (Input Neural Network)

                input_vector = processed_digit.reshape(784, 1)

                # Umpankan vektor input ke dalam model AI (proses Feedforward)
                output = net.feedforward(input_vector)
                outputs.append(output.flatten()) # Simpan array probabilitas untuk statistik chart

                # Mengambil indeks dengan nilai probabilitas tertinggi sebagai hasil tebakan angka (0-9)
                # PREDIKSI KELAS 0-35
                pred = int(np.argmax(output))

                # KONVERSI KE KARAKTER
                predictions.append(CLASS_NAMES[pred])

            # Jika semua kontur ternyata hanya noise/titik kecil dan tidak menghasilkan prediksi valid
            if len(predictions) == 0:
                st.warning("Please draw a digit first.")
            else:
                # Menggabungkan list teks prediksi menjadi satu kesatuan string (misal: ['1', '0'] -> "10")
                final_result = "".join(predictions)
                # Menampilkan visualisasi proses gambar AI di kolom kanan
                with right_col:
                    st.markdown("#### AI Visualization")
                    if display_images:
                        # Menggabungkan semua gambar 28x28 secara horizontal berdampingan
                        combined_img = np.hstack(display_images)
                        combined_img = (combined_img * 255).astype(np.uint8)

                        # Memperbesar gambar gabungan tersebut menggunakan metode INTER_NEAREST agar tidak blur/pecah
                        combined_img_large = cv2.resize(combined_img, (CANVAS_SIZE, int(CANVAS_SIZE / len(display_images))), interpolation=cv2.INTER_NEAREST)
                        st.image(combined_img_large, caption="Processed 28x28 Input per Digit", channels="RGB")
                # Tampilkan hasil teks tebakan final di bawah gambar
                st.markdown(f"## 🔍 Prediction: **{final_result}**")

                # Menampilkan statistik akurasi (Confidence Score) dalam bentuk grafik batang
                if outputs:
                    st.markdown("---")
                    st.markdown("### 📊 AI Confidence Stats")

                    # Membuat kolom dinamis sebanyak jumlah angka yang terdeteksi
                    chart_cols = st.columns(len(outputs))                   
                    # Iterasi untuk menampilkan chart probabilitas di tiap-tiap kolom angka
                    for idx, (col, out_data) in enumerate(zip(chart_cols, outputs)):
                        with col:
                            st.markdown(f"**Digit ke-{idx+1} (Prediksi: {predictions[idx]})**")
                            # Membuat diagram batang dari array output (indeks 0 sampai 9)
                            st.bar_chart(out_data, height=180)
    else:

        st.warning("Please draw a digit first.")