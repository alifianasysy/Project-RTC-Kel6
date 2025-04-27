use std::error::Error;
use std::fs::File;
use std::path::Path;
use csv::ReaderBuilder;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct FireData {
    #[serde(rename = "Temperature")]
    temperature: f64,
    #[serde(rename = "Humidity")]
    humidity: f64,
    #[serde(rename = "Gas_MQ3")]
    gas_mq3: f64,
    #[serde(rename = "Gas_MQ135")]
    gas_mq135: f64,
    #[serde(rename = "Fire_Risk")]
    fire_risk: u8,
}

#[derive(Serialize, Deserialize)]
pub struct Scaler {
    mins: Vec<f64>,
    maxs: Vec<f64>,
}

impl Scaler {
    pub fn new(data: &[(Vec<f64>, u8)]) -> Self {
        let features: Vec<Vec<f64>> = data.iter().map(|(f, _)| f.clone()).collect();
        let mut mins = vec![f64::INFINITY; features[0].len()];
        let mut maxs = vec![f64::NEG_INFINITY; features[0].len()];

        for feature_vec in features {
            for (i, &val) in feature_vec.iter().enumerate() {
                mins[i] = mins[i].min(val);
                maxs[i] = maxs[i].max(val);
            }
        }

        Self { mins, maxs }
    }

    pub fn normalize(&self, features: Vec<f64>) -> Vec<f64> {
        features.iter().enumerate().map(|(i, &val)| {
            let min = self.mins[i];
            let max = self.maxs[i];
            if max - min == 0.0 {
                0.0
            } else {
                (val - min) / (max - min)
            }
        }).collect()
    }

    #[allow(dead_code)]
    pub fn denormalize(&self, features: Vec<f64>) -> Vec<f64> {
        features.iter().enumerate().map(|(i, &val)| {
            let min = self.mins[i];
            let max = self.maxs[i];
            val * (max - min) + min
        }).collect()
    }
}

pub fn load_data(path: &Path) -> Result<Vec<FireData>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = ReaderBuilder::new().from_reader(file);
    let mut data = Vec::new();

    for result in rdr.deserialize() {
        let record: FireData = result?;
        data.push(record);
    }

    Ok(data)
}

pub fn process_data_advanced(data: &[FireData]) -> (Vec<(Vec<f64>, u8)>, Vec<usize>) {
    let features: Vec<Vec<f64>> = data.iter()
        .map(|d| vec![d.temperature, d.humidity, d.gas_mq3, d.gas_mq135])
        .collect();
    let labels: Vec<u8> = data.iter().map(|d| d.fire_risk).collect();

    let mut processed_data = Vec::new();
    let mut outlier_indices = Vec::new();

    for (i, (feature_vec, label)) in features.iter().zip(labels.iter()).enumerate() {
        let is_outlier = feature_vec.iter().any(|&val| {
            val < -1e6 || val > 1e6 || val.is_nan() || val.is_infinite()
        });

        if !is_outlier {
            processed_data.push((feature_vec.clone(), *label));
        } else {
            outlier_indices.push(i);
        }
    }

    (processed_data, outlier_indices)
}

pub fn normalize_data(data: &[(Vec<f64>, u8)]) -> (Vec<(Vec<f64>, u8)>, Scaler) {
    let scaler = Scaler::new(data);
    let normalized_data: Vec<(Vec<f64>, u8)> = data.iter().map(|(features, label)| {
        (scaler.normalize(features.clone()), *label)
    }).collect();
    (normalized_data, scaler)
}

pub fn create_sequences(data: &[(Vec<f64>, u8)], window_size: usize) -> Vec<(Vec<f64>, u8)> {
    let mut sequences = Vec::new();
    for i in 0..data.len().saturating_sub(window_size) {
        let mut sequence_features = Vec::new();
        for j in 0..window_size {
            sequence_features.extend(data[i + j].0.clone());
        }
        let target = data[i + window_size].1;
        sequences.push((sequence_features, target));
    }
    sequences
}