mod model;
mod data;
mod utils;

use std::error::Error;
use std::path::Path;
use std::fs::File;
use std::io::Write;
use std::env;
use model::{LSTMModel, ModelConfig};
use data::{load_data, normalize_data, create_sequences, process_data_advanced, Scaler};
use utils::plot_accuracy_vs_epochs;
use serde_json::json;

fn train_and_save() -> Result<(), Box<dyn Error>> {
    println!("Program Fire Detection with Neural Network - Training Mode");
    
    // Load dataset
    let data_path = Path::new("data/fire_detection.csv");
    let data = load_data(data_path)?;
    println!("Dataset berhasil dimuat. Jumlah data: {}", data.len());
    
    // Process data: handle outliers
    let (processed_data, _) = process_data_advanced(&data);
    println!("Pemrosesan data lanjutan selesai (deteksi dan penanganan outlier)");
    
    // Normalize data
    let (normalized_data, scaler) = normalize_data(&processed_data);
    let window_size = 3;
    let sequences = create_sequences(&normalized_data, window_size);
    
    // Calculate input size
    let feature_size = normalized_data[0].0.len();
    let input_size = window_size * feature_size;
    
    // Split data into training (20%), validation, and testing (80%)
    let total_sequences = sequences.len();
    let train_size = (total_sequences as f64 * 0.20) as usize; // 20% for training
    let val_size = (train_size as f64 * 0.20) as usize;        // 20% of training for validation
    
    let train_start = 0;
    let val_start = train_size - val_size;
    let test_start = train_size;
    
    let train_data = sequences[train_start..val_start].to_vec();
    let val_data = sequences[val_start..test_start].to_vec();
    let test_data = sequences[test_start..].to_vec();
    
    println!("\nData split complete:");
    println!("Data training: {} sequences", train_data.len());
    println!("Data validasi: {} sequences", val_data.len());
    println!("Data testing: {} sequences", test_data.len());
    
    // Model configuration
    let config = ModelConfig {
        input_size,
        hidden_size1: 32,
        hidden_size2: 16,
        output_size: 1,
        learning_rate: 0.01,
        epochs: 1000,
        decay_rate: 0.001,
        l2_reg: 0.0001,
    };
    
    // Train model with early stopping
    let mut model = LSTMModel::new(config);
    println!("\nMulai proses training model dengan early stopping...");
    let (train_accuracies, val_accuracies) = model.train_with_validation(&train_data, &val_data, 2000);
    println!("Training selesai!");
    
    // Compute highest training and validation accuracies
    let max_train_accuracy = train_accuracies.iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(&0.0) * 100.0;
    let max_val_accuracy = val_accuracies.iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(&0.0) * 100.0;
    
    println!("\nTraining Accuracy: {:.2}%", max_train_accuracy);
    println!("Validation Accuracy: {:.2}%", max_val_accuracy);
    
    // Remove the test accuracy evaluation and related print statements
    // Previously, this section included predictions, targets, binary_predictions, and accuracy calculation
    
    // Plot accuracy vs epochs
    plot_accuracy_vs_epochs(
        &train_accuracies,
        &val_accuracies,
        "Accuracy vs Epochs",
        "hasil_prediksi.png",
    )?;
    
    // Predict for the next sequence
    let last_sequence_idx = normalized_data.len() - window_size;
    let mut last_sequence_features = Vec::new();
    for i in 0..window_size {
        last_sequence_features.extend(normalized_data[last_sequence_idx + i].0.iter());
    }
    
    let threshold = 0.5;
    let next_prediction = model.predict_next(&last_sequence_features);
    let next_binary_prediction = if next_prediction >= threshold { 1 } else { 0 };
    println!("\nPrediksi risiko kebakaran berikutnya: {}", 
        if next_binary_prediction == 1 { "Ada risiko kebakaran" } else { "Tidak ada risiko kebakaran" });
    
    // Save the model and scaler
    let model_file = File::create("model.bin")?;
    bincode::serialize_into(model_file, &model)?;
    println!("Model disimpan ke 'model.bin'");

    let scaler_file = File::create("scaler.bin")?;
    bincode::serialize_into(scaler_file, &scaler)?;
    println!("Scaler disimpan ke 'scaler.bin'");
    
    // Save results to JSON, excluding test accuracy and predictions
    let results = json!({
        "max_train_accuracy": max_train_accuracy,
        "max_val_accuracy": max_val_accuracy,
        "next_prediction": next_binary_prediction
    });
    
    let mut file = File::create("results.json")?;
    file.write_all(serde_json::to_string_pretty(&results)?.as_bytes())?;
    println!("Hasil disimpan ke 'results.json'");
    
    Ok(())
}

fn predict_with_inputs() -> Result<(), Box<dyn Error>> {
    // Get command-line arguments (skip the program name)
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 5 || args[0] != "--predict" {
        let error_json = json!({
            "error": "Invalid arguments",
            "message": "Usage: ./fire_detection --predict <temperature> <humidity> <gas_mq3> <gas_mq135>"
        });
        println!("{}", serde_json::to_string(&error_json)?);
        return Ok(());
    }

    // Load the model and scaler
    let model: LSTMModel = match File::open("model.bin").map_err(|e| Box::<dyn Error>::from(e)).and_then(|file| {
        bincode::deserialize_from(file).map_err(|e| Box::<dyn Error>::from(std::io::Error::new(std::io::ErrorKind::Other, e)))
    }) {
        Ok(model) => {
            println!("Model berhasil dimuat dari 'model.bin'");
            model
        },
        Err(e) => {
            let error_json = json!({
                "error": "Failed to load model",
                "message": e.to_string()
            });
            println!("{}", serde_json::to_string(&error_json)?);
            return Ok(());
        }
    };

    let scaler: Scaler = match File::open("scaler.bin").map_err(|e| Box::<dyn Error>::from(e)).and_then(|file| {
        bincode::deserialize_from(file).map_err(|e| Box::<dyn Error>::from(std::io::Error::new(std::io::ErrorKind::Other, e)))
    }) {
        Ok(scaler) => {
            println!("Scaler berhasil dimuat dari 'scaler.bin'");
            scaler
        },
        Err(e) => {
            let error_json = json!({
                "error": "Failed to load scaler",
                "message": e.to_string()
            });
            println!("{}", serde_json::to_string(&error_json)?);
            return Ok(());
        }
    };

    // Parse the input parameters
    let temperature: f64 = match args[1].parse() {
        Ok(val) => val,
        Err(e) => {
            let error_json = json!({
                "error": "Invalid temperature",
                "message": e.to_string()
            });
            println!("{}", serde_json::to_string(&error_json)?);
            return Ok(());
        }
    };

    let humidity: f64 = match args[2].parse() {
        Ok(val) => val,
        Err(e) => {
            let error_json = json!({
                "error": "Invalid humidity",
                "message": e.to_string()
            });
            println!("{}", serde_json::to_string(&error_json)?);
            return Ok(());
        }
    };

    let gas_mq3: f64 = match args[3].parse() {
        Ok(val) => val,
        Err(e) => {
            let error_json = json!({
                "error": "Invalid gas_mq3",
                "message": e.to_string()
            });
            println!("{}", serde_json::to_string(&error_json)?);
            return Ok(());
        }
    };

    let gas_mq135: f64 = match args[4].parse() {
        Ok(val) => val,
        Err(e) => {
            let error_json = json!({
                "error": "Invalid gas_mq135",
                "message": e.to_string()
            });
            println!("{}", serde_json::to_string(&error_json)?);
            return Ok(());
        }
    };

    let window_size = 3;
    let mut input_features = Vec::new();
    // Repeat the input for the window size (to match training sequence format)
    for _ in 0..window_size {
        let features = vec![temperature, humidity, gas_mq3, gas_mq135];
        let normalized = scaler.normalize(features);
        input_features.extend(normalized);
    }

    // Make prediction
    let prediction = model.predict_next(&input_features);
    let threshold = 0.5;
    let binary_prediction = if prediction >= threshold { 1 } else { 0 };

    // Output the result in JSON format for the GUI to parse
    let result = json!({
        "prediction": binary_prediction,
        "probability": prediction
    });
    println!("{}", serde_json::to_string(&result)?);

    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let result = if args.len() > 1 && args[1] == "--predict" {
        predict_with_inputs()
    } else {
        train_and_save() 
    };

    if let Err(e) = result {
        let error_json = json!({
            "error": "Unexpected error",
            "message": e.to_string()
        });
        println!("{}", serde_json::to_string(&error_json).unwrap_or_else(|_| r#"{"error":"Failed to serialize error"}"#.to_string()));
    }
}