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

# fire_SVM
Berisi implementasi Machine Learning menggunakan Support Vector Machine (SVM) dan k-Nearest Neighbors (KNN) untuk Fire Detection. 

# trigonometri
Berisi program untuk perhitungan Deret Taylor dan Lookup Table menggunakan Rust. Program ini digunakan untuk perhitungan matematis yang efisien dan cepat, yang berguna dalam aplikasi yang memerlukan perhitungan trigonometrik yang cepat.



**🌟 Fitur Utama:**

**Pendekatan Multi-Model**: Proyek ini membandingkan dan menguji kinerja tiga algoritma machine learning (SVM, KNN, dan NN) untuk deteksi kebakaran

**Sensor Data Input**: Program ini menggunakan data dari sensor suhu, kelembaban, dan gas (MQ3 & MQ135) untuk mengidentifikasi potensi kebakaran di lingkungan.

**Prediksi Risiko Kebakaran**: Model ini menyediakan prediksi tentang kemungkinan terjadinya kebakaran (0 = Tidak ada kebakaran, 1 = Kebakaran terdeteksi), yang dapat digunakan dalam sistem keamanan atau aplikasi pemantauan.

**Scalable dan Extendable**: Struktur kode yang sederhana dan terorganisir memudahkan untuk proses pengembangan lebih lanjut.

**⚡ Mengapa Menggunakan Rust?**

Rust dipilih sebagai bahasa pemrograman karena keunggulannya dalam hal performa dan keamanan memori, yang sangat penting dalam aplikasi berbasis sensor dan sistem deteksi kebakaran yang memerlukan pengolahan data real-time. Dengan menggunakan Rust, program ini dapat memaksimalkan efisiensi proses pengelolaan data.

