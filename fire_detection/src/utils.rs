use std::error::Error;
use plotters::prelude::*;

pub fn calculate_accuracy(actual: &[u8], predicted: &[u8]) -> f64 {
    if actual.len() != predicted.len() || actual.is_empty() {
        return 0.0;
    }
    
    let correct: usize = actual.iter()
        .zip(predicted.iter())
        .filter(|(a, p)| a == p)
        .count();
    
    correct as f64 / actual.len() as f64
}

pub fn plot_accuracy_vs_epochs(
    train_accuracies: &[f64],
    val_accuracies: &[f64],
    title: &str,
    output_file: &str,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(output_file, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(
            0..train_accuracies.len() as i32,
            0f64..1.0
        )?;
    
    chart.configure_mesh()
        .x_desc("Epoch")
        .y_desc("Accuracy")
        .draw()?;
    
    chart.draw_series(LineSeries::new(
        (0..).zip(train_accuracies.iter()).map(|(x, &y)| (x as i32, y)),
        &BLUE,
    ))?
    .label("Train Accuracy")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
    
    chart.draw_series(LineSeries::new(
        (0..).zip(val_accuracies.iter()).map(|(x, &y)| (x as i32, y)),
        &RED,
    ))?
    .label("Val Accuracy")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
    
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;
    
    root.present()?;
    println!("Grafik akurasi vs epoch disimpan sebagai '{}'", output_file);
    Ok(())
}