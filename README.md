**🔥 Selamat datang di proyek Fire Detection!**

Repository ini berisi implementasi berbagai model pembelajaran mesin (Machine Learning/ML) yang dirancang untuk mendeteksi potensi kebakaran berdasarkan data sensor (Temperatur, Kelembapan, MQ3, MQ135). Program ini menggunakan beberapa metode yaitu, Support Vector Machine (SVM), k-Nearest Neighbors (KNN), dan Neural Networks (NN). Model ini dikembangkan dengan menggunakan bahasa pemrograman RUST, yang dikenal karena performanya yang tinggi dalam pengolahan data.

**🚒 Penjelasan File:**

**fire_detection**: Berisi program berbasis Neural Network (NN) untuk Fire Detection yang sudah terintegrasi dengan interface QT.

**fire_detection_nn**: Berisi program Neural Network (NN) untuk Fire Detection tanpa menggunakan antarmuka grafis (interface).

**fire_SVM**: Berisi implementasi Machine Learning menggunakan Support Vector Machine (SVM) dan k-Nearest Neighbors (KNN) untuk Fire Detection. 

**trigonometri**: Berisi program untuk perhitungan Deret Taylor dan Lookup Table menggunakan Rust. Program ini digunakan untuk perhitungan matematis yang efisien dan cepat, yang berguna dalam aplikasi yang memerlukan perhitungan trigonometrik yang cepat.

**🌟 Fitur Utama:**

**Pendekatan Multi-Model**: Proyek ini membandingkan dan menguji kinerja tiga algoritma machine learning (SVM, KNN, dan NN) untuk deteksi kebakaran

**Sensor Data Input**: Program ini menggunakan data dari sensor suhu, kelembaban, dan gas (MQ3 & MQ135) untuk mengidentifikasi potensi kebakaran di lingkungan.

**Prediksi Risiko Kebakaran**: Model ini menyediakan prediksi tentang kemungkinan terjadinya kebakaran (0 = Tidak ada kebakaran, 1 = Kebakaran terdeteksi), yang dapat digunakan dalam sistem keamanan atau aplikasi pemantauan.

**Scalable dan Extendable**: Struktur kode yang sederhana dan terorganisir memudahkan untuk proses pengembangan lebih lanjut.

**⚡ Mengapa Menggunakan Rust?**

Rust dipilih sebagai bahasa pemrograman karena keunggulannya dalam hal performa dan keamanan memori, yang sangat penting dalam aplikasi berbasis sensor dan sistem deteksi kebakaran yang memerlukan pengolahan data real-time. Dengan menggunakan Rust, program ini dapat memaksimalkan efisiensi proses pengelolaan data.

