use csv::ReaderBuilder;
use rand::seq::SliceRandom;
use rand::thread_rng;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::neighbors::knn_classifier::{KNNClassifier, KNNClassifierParameters};
use smartcore::metrics::accuracy;
use std::error::Error;

pub fn run_knn() -> Result<(), Box<dyn Error>> {
    // Baca dataset CSV
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path("data/fire_data.csv")?;

    let mut data: Vec<(Vec<f64>, u32)> = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let temperature: f64 = record[0].parse()?;
        let humidity: f64 = record[1].parse()?;
        let gas_mq3: f64 = record[2].parse()?;
        let gas_mq135: f64 = record[3].parse()?;
        let fire_risk: u32 = record[4].parse()?;

        data.push((vec![temperature, humidity, gas_mq3, gas_mq135], fire_risk));
    }

    // Acak data dan bagi menjadi training dan testing (80/20)
    let mut rng = thread_rng();
    data.shuffle(&mut rng);
    let split_index = (data.len() as f32 * 0.8) as usize;

    let (train_data, test_data) = data.split_at(split_index);

    let train_features: Vec<Vec<f64>> = train_data.iter().map(|(x, _)| x.clone()).collect();
    let train_targets: Vec<u32> = train_data.iter().map(|(_, y)| *y).collect();

    let test_features: Vec<Vec<f64>> = test_data.iter().map(|(x, _)| x.clone()).collect();
    let test_targets: Vec<u32> = test_data.iter().map(|(_, y)| *y).collect();

    // Training model KNN
    let x_train = DenseMatrix::from_2d_vec(&train_features)?;
    let knn_params = KNNClassifierParameters::default().with_k(3);
    let knn = KNNClassifier::fit(&x_train, &train_targets, knn_params)?;

    // Prediksi data testing
    let x_test = DenseMatrix::from_2d_vec(&test_features)?;
    let predictions = knn.predict(&x_test)?;
    let acc = accuracy(&test_targets, &predictions);
    println!("\nAkurasi Testing:\nkNN: {:.2}%", acc * 100.0);

    // Cetak hasil prediksi
    println!("\nHasil Prediksi (Data Testing):");
    for (i, sample) in test_features.iter().enumerate() {
        let pred = predictions[i];
        let label = if pred == 1 { "Kebakaran" } else { "Tidak Ada Kebakaran" };
        println!(
            "Data Testing {}: {:?} => kNN = {}",
            i + 1,
            sample,
            label
        );
    }

    Ok(())
}
