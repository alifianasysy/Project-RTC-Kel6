# 🔥 SELAMAT DATANG DI PROYEK FIRE DETECTION!

**SUPERVISOR : Ahmad Radhy, S.Si., M.Si**

**KELOMPOK 6 DEPARTEMEN TEKNIK INSTRUMENTASI - INSTITUT TEKNOLOGI SEPULUH NOPEMBER** 

**1. Alifian Asy Syifa (2042221137)**

**2. Rany Cahya Wijaya (2042221086)**

**3. Sherli Oktavianita (2042221101)**

Repository ini berisi implementasi berbagai model pembelajaran mesin (Machine Learning/ML) yang dirancang untuk mendeteksi potensi kebakaran berdasarkan data sensor (Temperatur, Kelembapan, MQ3, MQ135). Program ini menggunakan beberapa metode yaitu, Support Vector Machine (SVM), k-Nearest Neighbors (KNN), dan Neural Networks (NN). Model ini dikembangkan dengan menggunakan bahasa pemrograman RUST, yang dikenal karena performanya yang tinggi dalam pengolahan data.

# 📁 Daftar File, Deskripsi, dan Tata Cara Pembuatan

# fire_detection
Berisi program berbasis Neural Network (NN) untuk Fire Detection yang sudah terintegrasi dengan interface QT.

**📋 Langkah-Langkah:**

1. Tulis logika backend di Rust.

2. Hubungkan dengan antarmuka GUI berbasis Qt melalui Python.

3. Simpan sebagai fire_detection.rs.

**⚙️ Dependensi:**

1. Rust & Cargo

2. Python

3. PyQt5 (Instal via pip: pip install pyqt5)

**▶️ Cara Menjalankan:**

1. cargo build --release

2. Python qt_frontend.py

# fire_detection_nn 
Berisi program Neural Network (NN) untuk Fire Detection tanpa menggunakan antarmuka grafis (interface).

**📋 Langkah-Langkah:**

1. Mempersiapkan Dataset (Gunakan dataset dalam format CSV)

2. Inisialisasi Arsitektur Neural Network (Tentukan jumlah neuron pada input layer (sesuai jumlah fitur), hidden layer, dan output layer)

3. Bobot dan bias diinisialisasi secara acak menggunakan ndarray-rand
   
5. Gunakan fungsi sigmoid atau ReLU:

6. Hitung output dari setiap layer:

7. Lakukan training selama beberapa epoch, dan tampilkan error (loss) setiap epoch untuk melihat kemajuan:

8. Evaluasi dan Visualisasi

**⚙️ Dependensi:**

1. Rust & Cargo

2. ndarray 

3. ndarray-rand
   
4. plotters

**▶️ Cara Menjalankan:**

1. cargo build 

2. cargo run

# fire_SVM
Berisi implementasi Machine Learning menggunakan Support Vector Machine (SVM) dan k-Nearest Neighbors (KNN) untuk Fire Detection. 

**📋 Langkah-Langkah:**

1. Persiapan Dataset Fire Decetion (Gunakan Dataset dalam format CSV)

2. Buat model dengan Format smartcore (use smartcore::linalg::naive::dense_matrix::DenseMatrix;)

3. Implementasi KNN (Gunakan API smartcore untuk mengklasifikasikan dengan KNN)
   
   use smartcore::neighbors::knn_classifier::KNNClassifier;
   
   use smartcore::neighbors::knn_classifier::KNNClassifierParameters;

4. Implementasi SVM (Terapkan model SVM dengan kernel linier)

   use smartcore::svm::svc::SVC;
   
   use smartcore::svm::svc::SVCParameters;
   
   use smartcore::svm::Kernels;

5. Evaluasi Model

6. Hitung akurasi

7. Tampilkan Hasil Prediksi Interaktif (Untuk SVM terdapat Plot Grafik)

8. Bandingkan Performa KNN vs SVM

**⚙️ Dependensi:**

1. Rust & Cargo

2. smartcore
   
3. plotters

**▶️ Cara Menjalankan:**

1. cargo build 

2. cargo run


# trigonometri
Berisi program untuk perhitungan Deret Taylor dan Lookup Table menggunakan Rust. Program ini digunakan untuk perhitungan matematis yang efisien dan cepat, yang berguna dalam aplikasi yang memerlukan perhitungan trigonometrik yang cepat.

**📋 Langkah-Langkah:**

**Deret Taylor**

1. Tulis fungsi rekursif atau iteratif untuk menghitung nilai sin(x)/cos(x) berbasis deret tak hingga.

2. Uji dengan berbagai input x dan jumlah iterasi n.

3. deret_taylor.rs.
   
**Lookup Table**.

1. Buat array atau HashMap untuk menyimpan data numerik.

2. Tambahkan fungsi lookup() untuk mengambil nilai berdasarkan indeks/kunci.

3. Simpan sebagai lookup_table.rs.

**⚙️ Dependensi:**

1. Rust & Cargo

**▶️ Cara Menjalankan:**

1. cargo build

2. cargo run


# 🧭 Saran Alur Belajar
**📌 Jika kamu baru di Rust atau AI, ikuti urutan ini untuk hasil maksimal:**

1. Mulai dari file trigonmetri (Deret Taylor & Lookup Table) untuk dasar numerik

2. Pelajari SVM & KNN pada file fire_svm untuk dasar ML.

3. Eksplorasi file fire_detection_nn  untuk Neural Network.

4. Tutup dengan file fire_detection untuk integrasi GUI!


# 🌟 Fitur Utama:

**Pendekatan Multi-Model**: Proyek ini membandingkan dan menguji kinerja tiga algoritma machine learning (SVM, KNN, dan NN) untuk deteksi kebakaran

**Sensor Data Input**: Program ini menggunakan data dari sensor suhu, kelembaban, dan gas (MQ3 & MQ135) untuk mengidentifikasi potensi kebakaran di lingkungan.

**Prediksi Risiko Kebakaran**: Model ini menyediakan prediksi tentang kemungkinan terjadinya kebakaran (0 = Tidak ada kebakaran, 1 = Kebakaran terdeteksi), yang dapat digunakan dalam sistem keamanan atau aplikasi pemantauan.

**Scalable dan Extendable**: Struktur kode yang sederhana dan terorganisir memudahkan untuk proses pengembangan lebih lanjut.

# ⚡ Mengapa Menggunakan Rust?

Rust dipilih sebagai bahasa pemrograman karena keunggulannya dalam hal performa dan keamanan memori, yang sangat penting dalam aplikasi berbasis sensor dan sistem deteksi kebakaran yang memerlukan pengolahan data real-time. Dengan menggunakan Rust, program ini dapat memaksimalkan efisiensi proses pengelolaan data.

