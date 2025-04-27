use std::f64::consts::PI;

/// Fungsi untuk menghitung faktorial
fn factorial(n: u64) -> u64 {
    (1..=n).product()
}

/// Fungsi untuk menghitung sine(x) menggunakan deret Taylor
pub fn taylor_sin(x: f64, terms: u32) -> f64 {
    let mut sum = 0.0;
    for n in 0..terms {
        let exponent = 2 * n + 1;
        let term = ((-1.0f64).powi(n as i32) * x.powi(exponent as i32))
            / (factorial(exponent as u64) as f64);
        sum += term;
    }
    sum
}

/// Fungsi untuk menghitung cosine(x) menggunakan deret Taylor
pub fn taylor_cos(x: f64, terms: u32) -> f64 {
    let mut sum = 0.0;
    for n in 0..terms {
        let exponent = 2 * n;
        let term = ((-1.0f64).powi(n as i32) * x.powi(exponent as i32))
            / (factorial(exponent as u64) as f64);
        sum += term;
    }
    sum
}
