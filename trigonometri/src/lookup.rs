use std::f64::consts::PI;
use std::sync::LazyLock;

/// Jumlah titik data dalam lookup table
const TABLE_SIZE: usize = 91; // 0° sampai 90°

/// Lookup table sin dan cos dengan LazyLock
static SIN_TABLE: LazyLock<[f64; TABLE_SIZE]> = LazyLock::new(|| {
    let mut table = [0.0; TABLE_SIZE];
    for i in 0..TABLE_SIZE {
        table[i] = (i as f64 * PI / 180.0).sin();
    }
    table
});

static COS_TABLE: LazyLock<[f64; TABLE_SIZE]> = LazyLock::new(|| {
    let mut table = [0.0; TABLE_SIZE];
    for i in 0..TABLE_SIZE {
        table[i] = (i as f64 * PI / 180.0).cos();
    }
    table
});

/// Fungsi untuk lookup sin(x)
pub fn lookup_sin(degree: f64) -> f64 {
    let degree = degree.rem_euclid(360.0);
    let index = degree as usize;

    match index {
        0..=90 => SIN_TABLE[index],
        91..=180 => SIN_TABLE[180 - index],
        181..=270 => -SIN_TABLE[index - 180],
        _ => -SIN_TABLE[360 - index],
    }
}

/// Fungsi untuk lookup cos(x)
pub fn lookup_cos(degree: f64) -> f64 {
    let degree = degree.rem_euclid(360.0);
    let index = degree as usize;

    match index {
        0..=90 => COS_TABLE[index],
        91..=180 => -COS_TABLE[180 - index],
        181..=270 => -COS_TABLE[index - 180],
        _ => COS_TABLE[360 - index],
    }
}
