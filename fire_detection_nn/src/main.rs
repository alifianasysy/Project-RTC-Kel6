use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::error::Error;
use std::fs::File;
use csv::ReaderBuilder;
use plotters::prelude::*;
use std::io;

/// Fungsi aktivasi sigmoid
fn sigmoid(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

/// Turunan sigmoid
fn sigmoid_derivative(x: &Array2<f64>) -> Array2<f64> {
    x * &(1.0 - x)
}

/// Binary cross entropy loss
fn binary_cross_entropy(pred: &Array2<f64>, target: &Array2<f64>) -> f64 {
    let m = pred.len() as f64;
    let log_pred = pred.mapv(|v| (v + 1e-8).ln());
    let log_one_minus_pred = (1.0 - pred).mapv(|v| (v + 1e-8).ln());
    let loss = -(target * &log_pred + &(1.0 - target) * &log_one_minus_pred);
    loss.sum() / m
}

/// Load dataset dari CSV
fn load_csv(file_path: &str) -> Result<(Array2<f64>, Array2<f64>), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().from_reader(File::open(file_path)?);
    let headers = rdr.headers()?.clone();

    let feature_names = vec!["Temperature", "Humidity", "Gas_MQ3", "Gas_MQ135"];
    let label_name = "Fire_Risk";

    let feature_indices: Vec<usize> = feature_names
        .iter()
        .map(|name| headers.iter().position(|h| h == *name).unwrap())
        .collect();
    let label_index = headers.iter().position(|h| h == label_name).unwrap();

    let mut features = vec![];
    let mut labels = vec![];

    for result in rdr.records() {
        let record = result?;
        let mut feature_row = vec![];
        for &i in &feature_indices {
            let val = record.get(i).unwrap().trim();
            if val.is_empty() {
                continue;
            }
            feature_row.push(val.parse::<f64>()?);
        }

        let label_val = record.get(label_index).unwrap().trim();
        if label_val.is_empty() {
            continue;
        }
        features.extend(feature_row);
        labels.push(label_val.parse::<f64>()?);
    }

    let n_rows = labels.len();
    let feature_matrix = Array2::from_shape_vec((n_rows, feature_indices.len()), features)?;
    let label_matrix = Array2::from_shape_vec((n_rows, 1), labels)?;

    Ok((feature_matrix, label_matrix))
}

/// Bagi dataset: 80% train, 20% test
fn train_test_split(
    x: Array2<f64>,
    y: Array2<f64>,
    test_ratio: f64,
) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    let mut indices: Vec<usize> = (0..x.nrows()).collect();
    indices.shuffle(&mut thread_rng());

    let split = (x.nrows() as f64 * (1.0 - test_ratio)) as usize;
    let train_idx = &indices[..split];
    let test_idx = &indices[split..];

    let x_train = x.select(ndarray::Axis(0), train_idx);
    let y_train = y.select(ndarray::Axis(0), train_idx);
    let x_test = x.select(ndarray::Axis(0), test_idx);
    let y_test = y.select(ndarray::Axis(0), test_idx);

    (x_train, y_train, x_test, y_test)
}

/// Prediksi untuk input manual
fn predict(input: &Array2<f64>, w1: &Array2<f64>, w2: &Array2<f64>) -> f64 {
    let z1 = input.dot(w1);
    let a1 = sigmoid(&z1);
    let z2 = a1.dot(w2);
    let a2 = sigmoid(&z2);
    a2[[0, 0]]
}

/// Fungsi untuk plot loss & accuracy
fn plot_loss_accuracy(
    losses: &[f64],
    accuracies: &[f64],
    title: &str,
    loss_file: &str,
    acc_file: &str,
) -> Result<(), Box<dyn Error>> {
    // Loss
    let root_loss = BitMapBackend::new(loss_file, (600, 400)).into_drawing_area();
    root_loss.fill(&WHITE)?;
    let max_loss = *losses.iter().fold(&0.0, |a, b| if a > b { a } else { b });
    let mut chart_loss = ChartBuilder::on(&root_loss)
        .caption(format!("{} Loss", title), ("Arial", 20).into_font())
        .build_cartesian_2d(0..losses.len() as i32, 0f64..max_loss)?;
    chart_loss.configure_mesh().x_desc("Epoch").y_desc("Loss").draw()?;
    chart_loss.draw_series(LineSeries::new(
        (0..).zip(losses.iter()).map(|(x, &y)| (x as i32, y)),
        &RED,
    ))?;
    root_loss.present()?;

    // Accuracy
    let root_acc = BitMapBackend::new(acc_file, (600, 400)).into_drawing_area();
    root_acc.fill(&WHITE)?;
    let mut chart_acc = ChartBuilder::on(&root_acc)
        .caption(format!("{} Accuracy", title), ("Arial", 20).into_font())
        .build_cartesian_2d(0..accuracies.len() as i32, 0f64..1.0)?;
    chart_acc.configure_mesh().x_desc("Epoch").y_desc("Accuracy").draw()?;
    chart_acc.draw_series(LineSeries::new(
        (0..).zip(accuracies.iter()).map(|(x, &y)| (x as i32, y)),
        &BLUE,
    ))?;
    root_acc.present()?;

    println!("Grafik {} tersimpan: {} & {}", title, loss_file, acc_file);
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    // Load dan split dataset
    let (x, y) = load_csv("fire_data.csv")?;
    let (x_train, y_train, x_test, y_test) = train_test_split(x, y, 0.20);

    // Inisialisasi parameter
    let input_size = x_train.ncols();
    let hidden_size = 64;
    let output_size = 1;
    let learning_rate = 0.0001;
    let epochs = 2000;

    let mut w1 = Array2::random((input_size, hidden_size), Uniform::new(-1.0, 1.0));
    let mut w2 = Array2::random((hidden_size, output_size), Uniform::new(-1.0, 1.0));

    let mut train_losses = Vec::new();
    let mut train_accuracies = Vec::new();
    let mut test_losses = Vec::new();
    let mut test_accuracies = Vec::new();

    // Training loop
    for epoch in 0..epochs {
        let z1 = x_train.dot(&w1);
        let a1 = sigmoid(&z1);
        let z2 = a1.dot(&w2);
        let a2 = sigmoid(&z2);

        let loss = binary_cross_entropy(&a2, &y_train);
        train_losses.push(loss);

        let dz2 = &a2 - &y_train;
        let dw2 = a1.t().dot(&dz2) / x_train.nrows() as f64;
        let dz1 = dz2.dot(&w2.t()) * sigmoid_derivative(&a1);
        let dw1 = x_train.t().dot(&dz1) / x_train.nrows() as f64;

        w2 -= &(dw2 * learning_rate);
        w1 -= &(dw1 * learning_rate);

        let pred = sigmoid(&sigmoid(&x_train.dot(&w1)).dot(&w2));
        let predicted = pred.mapv(|v| if v > 0.5 { 1.0 } else { 0.0 });
        let correct = predicted
            .iter()
            .zip(y_train.iter())
            .filter(|(a, b)| (*a - *b).abs() < 1e-6)
            .count();
        let acc = correct as f64 / y_train.nrows() as f64;
        train_accuracies.push(acc);

        let test_pred = sigmoid(&sigmoid(&x_test.dot(&w1)).dot(&w2));
        let test_loss = binary_cross_entropy(&test_pred, &y_test);
        let test_pred_label = test_pred.mapv(|v| if v > 0.5 { 1.0 } else { 0.0 });
        let test_correct = test_pred_label
            .iter()
            .zip(y_test.iter())
            .filter(|(a, b)| (*a - *b).abs() < 1e-6)
            .count();
        let test_acc = test_correct as f64 / y_test.nrows() as f64;
        test_losses.push(test_loss);
        test_accuracies.push(test_acc);

        if epoch % 100 == 0 {
            println!(
                "Epoch {}: Train Loss = {:.4}, Train Acc = {:.2}%, Test Loss = {:.4}, Test Acc = {:.2}%",
                epoch,
                loss,
                acc * 100.0,
                test_loss,
                test_acc * 100.0
            );
        }
    }

    // Tampilkan akurasi akhir
    println!("\n=== Final Evaluation ===");
    println!(
        "Final Train Accuracy: {:.2}%",
        *train_accuracies.last().unwrap() * 100.0
    );
    println!(
        "Final Test Accuracy:  {:.2}%",
        *test_accuracies.last().unwrap() * 100.0
    );

    // Plot grafik
    plot_loss_accuracy(
        &train_losses,
        &train_accuracies,
        "Training",
        "training_loss.png",
        "training_accuracy.png",
    )?;
    plot_loss_accuracy(
        &test_losses,
        &test_accuracies,
        "Testing",
        "testing_loss.png",
        "testing_accuracy.png",
    )?;

    // Manual Testing
    println!("\n=== Manual Testing ===");
    'outer: loop {
        println!("\nMasukkan data sensor satu per satu:");

        let mut values = vec![];
        let sensor_names = ["Temperature", "Humidity", "Gas_MQ3", "Gas_MQ135"];

        for sensor in sensor_names.iter() {
            loop {
                println!("Masukkan nilai untuk {} (atau 'keluar' untuk selesai):", sensor);
                let mut input = String::new();
                io::stdin().read_line(&mut input)?;
                let input = input.trim();

                if input.to_lowercase() == "keluar" {
                    println!("Selesai.");
                    break 'outer;
                }

                match input.parse::<f64>() {
                    Ok(value) => {
                        values.push(value);
                        break;
                    }
                    Err(_) => {
                        println!("Error: Masukkan nilai numerik yang valid.");
                    }
                }
            }
        }

        let input_array = Array2::from_shape_vec((1, 4), values.clone())?;
        let prediction = predict(&input_array, &w1, &w2);
        let result = if prediction > 0.5 {
            "Fire Risk Detected"
        } else {
            "No Fire Risk"
        };

        println!("\nInput:");
        for (i, &value) in values.iter().enumerate() {
            println!("{}: {:.1}", sensor_names[i], value);
        }
        println!("Prediction: {}", result);

        println!("\nLanjutkan testing? (y/n)");
        let mut choice = String::new();
        io::stdin().read_line(&mut choice)?;
        let choice = choice.trim().to_lowercase();

        if choice == "n" {
            println!("Selesai.");
            break;
        } else if choice != "y" {
            println!("Pilihan tidak valid, dianggap selesai.");
            break;
        }
    }

    Ok(())
}
