mod taylor;
mod lookup;

use taylor::{taylor_sin, taylor_cos};
use lookup::{lookup_sin, lookup_cos};
use std::f64::consts::PI;
use std::io;

fn main() {
    loop {
        println!("Masukkan sudut dalam derajat (atau ketik 'selesai' untuk keluar):");

        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("Gagal membaca input");

        let input = input.trim();

        // Cek apakah user mau selesai
        if input.eq_ignore_ascii_case("selesai") {
            println!("Program selesai. Terima kasih!");
            break;
        }

        // Coba parsing ke angka
        let angle_deg: f64 = match input.parse() {
            Ok(num) => num,
            Err(_) => {
                println!("Input tidak valid. Masukkan angka atau ketik 'selesai'!");
                continue;
            }
        };

        let angle_rad = angle_deg * PI / 180.0;
        let terms = 10;

        // Taylor Series
        let sin_taylor = taylor_sin(angle_rad, terms);
        let cos_taylor = taylor_cos(angle_rad, terms);

        // Lookup Table
        let sin_lookup = lookup_sin(angle_deg);
        let cos_lookup = lookup_cos(angle_deg);

        println!("\n--- Hasil Taylor Series ---");
        println!("sin({}) ≈ {}", angle_deg, sin_taylor);
        println!("cos({}) ≈ {}", angle_deg, cos_taylor);

        println!("\n--- Hasil Lookup Table ---");
        println!("sin({}) ≈ {}", angle_deg, sin_lookup);
        println!("cos({}) ≈ {}", angle_deg, cos_lookup);
        println!(); // Biar ada spasi
    }
}
