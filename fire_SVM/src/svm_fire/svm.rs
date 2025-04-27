use csv::ReaderBuilder;
use rand::seq::SliceRandom;
use rand::thread_rng;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::svm::svc::{SVC, SVCParameters};
use smartcore::svm::Kernels;
use smartcore::metrics::accuracy;
use plotters::prelude::*;
use std::error::Error;

pub fn run_svm_fire_classification() -> Result<(), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path("data/fire_data.csv")?;

    let mut data: Vec<(Vec<f64>, i32)> = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let temperature: f64 = record[0].parse()?;
        let humidity: f64 = record[1].parse()?;
        let gas_mq3: f64 = record[2].parse()?;
        let gas_mq135: f64 = record[3].parse()?;
        let fire_risk: i32 = record[4].parse()?;

        let label = if fire_risk == 1 { 1 } else { -1 };
        data.push((vec![temperature, humidity, gas_mq3, gas_mq135], label));
    }

    // Acak data dan bagi menjadi training dan testing (80/20)
    let mut rng = thread_rng();
    data.shuffle(&mut rng);
    let split_index = (data.len() as f32 * 0.8) as usize;

    let (train_data, test_data) = data.split_at(split_index);

    let train_features: Vec<Vec<f64>> = train_data.iter().map(|(x, _)| x.clone()).collect();
    let train_targets: Vec<i32> = train_data.iter().map(|(_, y)| *y).collect();

    let test_features: Vec<Vec<f64>> = test_data.iter().map(|(x, _)| x.clone()).collect();
    let test_targets: Vec<i32> = test_data.iter().map(|(_, y)| *y).collect();

    let x_train = DenseMatrix::from_2d_vec(&train_features)?;
    let x_test = DenseMatrix::from_2d_vec(&test_features)?;

    let params = SVCParameters::default()
        .with_c(1.0)
        .with_kernel(Kernels::rbf().with_gamma(0.1));

    let model = SVC::fit(&x_train, &train_targets, &params)?;
    let predictions = model.predict(&x_test)?;
    let predictions_i32: Vec<i32> = predictions.iter().map(|&p| p as i32).collect();

    let acc = accuracy(&test_targets, &predictions_i32);
    println!("Akurasi model (testing): {:.2}%", acc * 100.0);

    // Plot data testing
    let root = BitMapBackend::new("output/fire_classification.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Klasifikasi Kebakaran (Data Testing)", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(20f64..40f64, 0f64..250f64)?;

    chart.configure_mesh().draw()?;

    for i in 0..test_features.len() {
        let x_val = test_features[i][0]; // Temperature
        let y_val = test_features[i][2]; // Gas MQ3
        let label = predictions_i32[i];
        let color = if label == 1 { &RED } else { &BLUE };
        chart.draw_series(std::iter::once(Circle::new((x_val, y_val), 4, color.filled())))?;
    }

    println!("Plot klasifikasi disimpan ke output/fire_classification.png");
    Ok(())
}
