mod knn_fire;
mod svm_fire;

fn main() {
    println!("=== Pendeteksi Kebakaran ===\n");

    // Jalankan KNN
    println!("--- KNN CLASSIFIER ---");
    if let Err(e) = knn_fire::knn::run_knn() {
        eprintln!("Terjadi kesalahan di KNN: {}", e);
    }

    // Jalankan SVM
    println!("\n--- SVM CLASSIFIER ---");
    if let Err(e) = svm_fire::svm::run_svm_fire_classification() {
        eprintln!("Terjadi kesalahan di SVM: {}", e);
    }
}
