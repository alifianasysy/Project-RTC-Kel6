use ndarray::{Array1, Array2, Axis};
use rand_distr::{Normal, Distribution};
use rand::thread_rng;
use rand::seq::SliceRandom;
use serde::{Serialize, Deserialize};
use crate::utils::calculate_accuracy;

#[derive(Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub input_size: usize,
    pub hidden_size1: usize,
    pub hidden_size2: usize,
    pub output_size: usize,
    pub learning_rate: f64,
    pub epochs: usize,
    pub decay_rate: f64,
    pub l2_reg: f64,
}

#[derive(Serialize, Deserialize)]
struct LSTMCell {
    w_i: Array2<f64>,
    u_i: Array2<f64>,
    b_i: Array1<f64>,
    w_f: Array2<f64>,
    u_f: Array2<f64>,
    b_f: Array1<f64>,
    w_o: Array2<f64>,
    u_o: Array2<f64>,
    b_o: Array1<f64>,
    w_c: Array2<f64>,
    u_c: Array2<f64>,
    b_c: Array1<f64>,
}

impl LSTMCell {
    fn new(input_size: usize, hidden_size: usize) -> Self {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 0.1).unwrap();
        
        let w_i = Array2::from_shape_fn((hidden_size, input_size), |_| normal.sample(&mut rng));
        let u_i = Array2::from_shape_fn((hidden_size, hidden_size), |_| normal.sample(&mut rng));
        let b_i = Array1::zeros(hidden_size);
        
        let w_f = Array2::from_shape_fn((hidden_size, input_size), |_| normal.sample(&mut rng));
        let u_f = Array2::from_shape_fn((hidden_size, hidden_size), |_| normal.sample(&mut rng));
        let b_f = Array1::ones(hidden_size);
        
        let w_o = Array2::from_shape_fn((hidden_size, input_size), |_| normal.sample(&mut rng));
        let u_o = Array2::from_shape_fn((hidden_size, hidden_size), |_| normal.sample(&mut rng));
        let b_o = Array1::zeros(hidden_size);
        
        let w_c = Array2::from_shape_fn((hidden_size, input_size), |_| normal.sample(&mut rng));
        let u_c = Array2::from_shape_fn((hidden_size, hidden_size), |_| normal.sample(&mut rng));
        let b_c = Array1::zeros(hidden_size);
        
        Self {
            w_i, u_i, b_i,
            w_f, u_f, b_f,
            w_o, u_o, b_o,
            w_c, u_c, b_c,
        }
    }
    
    fn forward(&self, x: &Array1<f64>, h_prev: &Array1<f64>, c_prev: &Array1<f64>) -> (Array1<f64>, Array1<f64>, ForwardCache) {
        let i_gate = self.w_i.dot(x) + self.u_i.dot(h_prev) + &self.b_i;
        let i = sigmoid(&i_gate);
        
        let f_gate = self.w_f.dot(x) + self.u_f.dot(h_prev) + &self.b_f;
        let f = sigmoid(&f_gate);
        
        let o_gate = self.w_o.dot(x) + self.u_o.dot(h_prev) + &self.b_o;
        let o = sigmoid(&o_gate);
        
        let c_tilde_gate = self.w_c.dot(x) + self.u_c.dot(h_prev) + &self.b_c;
        let c_tilde = tanh(&c_tilde_gate);
        
        let c = &f * c_prev + &i * &c_tilde;
        let h = &o * tanh(&c);
        
        let cache = ForwardCache {
            x: x.clone(),
            h_prev: h_prev.clone(),
            c_prev: c_prev.clone(),
            i, f, o, c_tilde,
            c: c.clone(),
            i_gate, f_gate, o_gate, c_tilde_gate,
        };
        
        (h, c, cache)
    }
    
    fn backward(
        &mut self, 
        dh_next: &Array1<f64>, 
        dc_next: &Array1<f64>, 
        cache: &ForwardCache,
        learning_rate: f64
    ) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let c_prev_tanh = tanh(&cache.c);
        
        let do_gate = dh_next * &c_prev_tanh;
        let do_gate = do_gate * sigmoid_derivative(&cache.o_gate);
        
        let dc = dc_next + dh_next * &cache.o * tanh_derivative(&cache.c);
        
        let df_gate = &dc * &cache.c_prev;
        let df_gate = df_gate * sigmoid_derivative(&cache.f_gate);
        
        let di_gate = &dc * &cache.c_tilde;
        let di_gate = di_gate * sigmoid_derivative(&cache.i_gate);
        
        let dc_tilde_gate = &dc * &cache.i;
        let dc_tilde_gate = dc_tilde_gate * tanh_derivative(&cache.c_tilde_gate);
        
        let dw_i = di_gate.clone().insert_axis(Axis(1)).dot(&cache.x.clone().insert_axis(Axis(0)));
        let du_i = di_gate.clone().insert_axis(Axis(1)).dot(&cache.h_prev.clone().insert_axis(Axis(0)));
        let db_i = di_gate.clone();
        
        let dw_f = df_gate.clone().insert_axis(Axis(1)).dot(&cache.x.clone().insert_axis(Axis(0)));
        let du_f = df_gate.clone().insert_axis(Axis(1)).dot(&cache.h_prev.clone().insert_axis(Axis(0)));
        let db_f = df_gate.clone();
        
        let dw_o = do_gate.clone().insert_axis(Axis(1)).dot(&cache.x.clone().insert_axis(Axis(0)));
        let du_o = do_gate.clone().insert_axis(Axis(1)).dot(&cache.h_prev.clone().insert_axis(Axis(0)));
        let db_o = do_gate.clone();
        
        let dw_c = dc_tilde_gate.clone().insert_axis(Axis(1)).dot(&cache.x.clone().insert_axis(Axis(0)));
        let du_c = dc_tilde_gate.clone().insert_axis(Axis(1)).dot(&cache.h_prev.clone().insert_axis(Axis(0)));
        let db_c = dc_tilde_gate.clone();
        
        self.w_i -= &(dw_i * learning_rate);
        self.u_i -= &(du_i * learning_rate);
        self.b_i -= &(db_i * learning_rate);
        
        self.w_f -= &(dw_f * learning_rate);
        self.u_f -= &(du_f * learning_rate);
        self.b_f -= &(db_f * learning_rate);
        
        self.w_o -= &(dw_o * learning_rate);
        self.u_o -= &(du_o * learning_rate);
        self.b_o -= &(db_o * learning_rate);
        
        self.w_c -= &(dw_c * learning_rate);
        self.u_c -= &(du_c * learning_rate);
        self.b_c -= &(db_c * learning_rate);
        
        let dx = self.w_i.t().dot(&di_gate) + 
                 self.w_f.t().dot(&df_gate) + 
                 self.w_o.t().dot(&do_gate) + 
                 self.w_c.t().dot(&dc_tilde_gate);
                 
        let dh_prev = self.u_i.t().dot(&di_gate) + 
                      self.u_f.t().dot(&df_gate) + 
                      self.u_o.t().dot(&do_gate) + 
                      self.u_c.t().dot(&dc_tilde_gate);
                      
        let dc_prev = &cache.f * dc;
        
        (dx, dh_prev, dc_prev)
    }
}

#[derive(Serialize, Deserialize)]
struct ForwardCache {
    x: Array1<f64>,
    h_prev: Array1<f64>,
    c_prev: Array1<f64>,
    i: Array1<f64>,
    f: Array1<f64>,
    o: Array1<f64>,
    c_tilde: Array1<f64>,
    c: Array1<f64>,
    i_gate: Array1<f64>,
    f_gate: Array1<f64>,
    o_gate: Array1<f64>,
    c_tilde_gate: Array1<f64>,
}

#[derive(Serialize, Deserialize)]
struct DenseLayer {
    weights: Array2<f64>,
    bias: Array1<f64>,
}

impl DenseLayer {
    fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 0.1).unwrap();
        
        let weights = Array2::from_shape_fn((output_size, input_size), |_| normal.sample(&mut rng));
        let bias = Array1::zeros(output_size);
        
        Self { weights, bias }
    }
    
    fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        &self.weights.dot(x) + &self.bias
    }
    
    fn backward(&mut self, x: &Array1<f64>, grad_output: &Array1<f64>, learning_rate: f64) -> Array1<f64> {
        let dweights = grad_output.clone().insert_axis(Axis(1)).dot(&x.clone().insert_axis(Axis(0)));
        let dbias = grad_output.clone();
        
        self.weights -= &(dweights * learning_rate);
        self.bias -= &(dbias * learning_rate);
        
        self.weights.t().dot(grad_output)
    }
}

#[derive(Serialize, Deserialize)]
pub struct LSTMModel {
    config: ModelConfig,
    lstm_layer1: LSTMCell,
    lstm_layer2: LSTMCell,
    output_layer: DenseLayer,
}

impl LSTMModel {
    pub fn new(config: ModelConfig) -> Self {
        let lstm_layer1 = LSTMCell::new(config.input_size, config.hidden_size1);
        let lstm_layer2 = LSTMCell::new(config.hidden_size1, config.hidden_size2);
        let output_layer = DenseLayer::new(config.hidden_size2, config.output_size);
        
        Self { config, lstm_layer1, lstm_layer2, output_layer }
    }
    
    pub fn train(&mut self, train_data: &[(Vec<f64>, u8)]) {
        let initial_lr = self.config.learning_rate;
        let decay_rate = self.config.decay_rate;
        
        for epoch in 0..self.config.epochs {
            let learning_rate = initial_lr / (1.0 + decay_rate * epoch as f64);
            let mut total_loss = 0.0;
            
            let mut shuffled_data = train_data.to_vec();
            shuffled_data.shuffle(&mut thread_rng());
            
            for (seq, target) in &shuffled_data {
                let x = Array1::from_vec(seq.clone());
                let target_val = *target as f64;
                
                let h1 = Array1::zeros(self.config.hidden_size1);
                let c1 = Array1::zeros(self.config.hidden_size1);
                let h2 = Array1::zeros(self.config.hidden_size2);
                let c2 = Array1::zeros(self.config.hidden_size2);
                
                let (new_h1, _new_c1, cache1) = self.lstm_layer1.forward(&x, &h1, &c1);
                let (new_h2, _new_c2, cache2) = self.lstm_layer2.forward(&new_h1, &h2, &c2);
                
                let output = self.output_layer.forward(&new_h2);
                let prediction = sigmoid(&output)[0];
                
                let l2_term = self.config.l2_reg * sum_of_squares(&self.output_layer.weights);
                let loss = -(target_val * (prediction as f64).ln() + (1.0 - target_val) * (1.0 - prediction as f64).ln()) + l2_term;
                total_loss += loss;
                
                let doutput = Array1::from_vec(vec![prediction - target_val]);
                let dh2 = self.output_layer.backward(&new_h2, &doutput, learning_rate);
                
                let dc2_next = Array1::zeros(self.config.hidden_size2);
                let (dh1, _, _) = self.lstm_layer2.backward(&dh2, &dc2_next, &cache2, learning_rate);
                
                let dc1_next = Array1::zeros(self.config.hidden_size1);
                let (_, _, _) = self.lstm_layer1.backward(&dh1, &dc1_next, &cache1, learning_rate);
            }
            
            if epoch % 100 == 0 {
                let avg_loss = total_loss / train_data.len() as f64;
                println!("Epoch {}: Loss = {:.6}, LR = {:.6}", epoch, avg_loss, learning_rate);
            }
        }
    }

    pub fn train_with_validation(
        &mut self,
        train_data: &[(Vec<f64>, u8)],
        val_data: &[(Vec<f64>, u8)],
        patience: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let initial_lr = self.config.learning_rate;
        let decay_rate = self.config.decay_rate;
        let mut best_val_accuracy = 0.0;
        let mut epochs_without_improvement = 0;
    
        let mut train_accuracies = Vec::new();
        let mut val_accuracies = Vec::new();
    
        for epoch in 0..self.config.epochs {
            let learning_rate = initial_lr / (1.0 + decay_rate * epoch as f64);
            let mut total_loss = 0.0;
    
            let mut shuffled_data = train_data.to_vec();
            shuffled_data.shuffle(&mut thread_rng());
    
            for (seq, target) in &shuffled_data {
                let x = Array1::from_vec(seq.clone());
                let target_val = *target as f64;
    
                let h1 = Array1::zeros(self.config.hidden_size1);
                let c1 = Array1::zeros(self.config.hidden_size1);
                let h2 = Array1::zeros(self.config.hidden_size2);
                let c2 = Array1::zeros(self.config.hidden_size2);
    
                let (new_h1, _new_c1, cache1) = self.lstm_layer1.forward(&x, &h1, &c1);
                let (new_h2, _new_c2, cache2) = self.lstm_layer2.forward(&new_h1, &h2, &c2);
                let output = self.output_layer.forward(&new_h2);
                let prediction = sigmoid(&output)[0];
    
                let l2_term = self.config.l2_reg * sum_of_squares(&self.output_layer.weights);
                let loss = -(target_val * (prediction as f64).ln() + (1.0 - target_val) * (1.0 - prediction as f64).ln()) + l2_term;
                total_loss += loss;
    
                let doutput = Array1::from_vec(vec![prediction - target_val]);
                let dh2 = self.output_layer.backward(&new_h2, &doutput, learning_rate);
                let dc2_next = Array1::zeros(self.config.hidden_size2);
                let (dh1, _, _) = self.lstm_layer2.backward(&dh2, &dc2_next, &cache2, learning_rate);
                let dc1_next = Array1::zeros(self.config.hidden_size1);
                let (_, _, _) = self.lstm_layer1.backward(&dh1, &dc1_next, &cache1, learning_rate);
            }
    
            // Compute training accuracy
            let train_predictions = self.predict(train_data);
            let train_targets: Vec<u8> = train_data.iter().map(|(_, target)| *target).collect();
            let train_binary_predictions: Vec<u8> = train_predictions.iter()
                .map(|&p| if p >= 0.5 { 1 } else { 0 })
                .collect();
            let train_accuracy = calculate_accuracy(&train_targets, &train_binary_predictions);
            train_accuracies.push(train_accuracy);
    
            // Compute validation accuracy
            let val_predictions = self.predict(val_data);
            let val_targets: Vec<u8> = val_data.iter().map(|(_, target)| *target).collect();
            let val_binary_predictions: Vec<u8> = val_predictions.iter()
                .map(|&p| if p >= 0.5 { 1 } else { 0 })
                .collect();
            let val_accuracy = calculate_accuracy(&val_targets, &val_binary_predictions);
            val_accuracies.push(val_accuracy);
    
            if epoch % 100 == 0 {
                println!("Epoch {}: Train Loss = {:.6}, Train Accuracy = {:.2}%, Val Accuracy = {:.2}%", 
                    epoch, total_loss / train_data.len() as f64, train_accuracy * 100.0, val_accuracy * 100.0);
            }
            
            if val_accuracy > best_val_accuracy {
                best_val_accuracy = val_accuracy;
                epochs_without_improvement = 0;
            } else {
                epochs_without_improvement += 1;
                if epochs_without_improvement >= patience {
                    println!("Early stopping at epoch {}", epoch);
                    break;
                }
            }
        }
    
        (train_accuracies, val_accuracies)
    }
    
    pub fn predict(&self, test_data: &[(Vec<f64>, u8)]) -> Vec<f64> {
        let mut predictions = Vec::new();
        
        for (seq, _) in test_data {
            let x = Array1::from_vec(seq.clone());
            
            let h1 = Array1::zeros(self.config.hidden_size1);
            let c1 = Array1::zeros(self.config.hidden_size1);
            let h2 = Array1::zeros(self.config.hidden_size2);
            let c2 = Array1::zeros(self.config.hidden_size2);
            
            let (new_h1, _new_c1, _) = self.lstm_layer1.forward(&x, &h1, &c1);
            let (new_h2, _new_c2, _) = self.lstm_layer2.forward(&new_h1, &h2, &c2);
            
            let output = self.output_layer.forward(&new_h2);
            let prediction = sigmoid(&output)[0];
            
            predictions.push(prediction);
        }
        
        predictions
    }
    
    pub fn predict_next(&self, sequence: &[f64]) -> f64 {
        let x = Array1::from_vec(sequence.to_vec());
        
        let h1 = Array1::zeros(self.config.hidden_size1);
        let c1 = Array1::zeros(self.config.hidden_size1);
        let h2 = Array1::zeros(self.config.hidden_size2);
        let c2 = Array1::zeros(self.config.hidden_size2);
        
        let (new_h1, _new_c1, _) = self.lstm_layer1.forward(&x, &h1, &c1);
        let (new_h2, _new_c2, _) = self.lstm_layer2.forward(&new_h1, &h2, &c2);
        
        let output = self.output_layer.forward(&new_h2);
        sigmoid(&output)[0]
    }
}

fn sigmoid(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

fn sigmoid_derivative(x: &Array1<f64>) -> Array1<f64> {
    let sig = sigmoid(x);
    &sig * &(1.0 - &sig)
}

fn tanh(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| v.tanh())
}

fn tanh_derivative(x: &Array1<f64>) -> Array1<f64> {
    let t = tanh(x);
    1.0 - &(&t * &t)
}

fn sum_of_squares(arr: &Array2<f64>) -> f64 {
    arr.iter().map(|&x| x.powi(2)).sum()
}