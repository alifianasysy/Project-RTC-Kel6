import sys
import json
import subprocess
import pandas as pd
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QTableView, QLabel,
                             QTabWidget, QHeaderView, QToolBar, QLineEdit, QProgressBar,
                             QAction, QMessageBox, QScrollArea, QFormLayout, QDoubleSpinBox)
from PyQt5.QtGui import QIcon, QFont, QPixmap
from PyQt5.QtCore import Qt, QSortFilterProxyModel
from PyQt5.Qt import QStandardItemModel, QStandardItem

class FireDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fire Detection App")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(800, 600)

        # Apply modern stylesheet with black background
        self.setStyleSheet("""
            QMainWindow {
                background-color: #000000;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ff6b6b, stop:1 #ee5253);
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-family: 'Roboto', 'Arial', sans-serif;
                font-size: 14px;
                font-weight: bold;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ff8787, stop:1 #ff6b6b);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ee5253, stop:1 #d43f3f);
                box-shadow: 0 1px 2px rgba(0,0,0,0.3);
            }
            QTabWidget::pane {
                border: 1px solid #e0e4e8;
                background: #ffffff;
                border-radius: 8px;
            }
            QTabBar::tab {
                background: #e8ecef;
                color: #333;
                padding: 12px 20px;
                margin-right: 4px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-family: 'Roboto', 'Arial', sans-serif;
                font-size: 14px;
                font-weight: 500;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ff6b6b, stop:1 #ee5253);
                color: white;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background: #d1d5da;
            }
            QTableView {
                gridline-color: #e0e4e8;
                background-color: #ffffff;
                font-family: 'Roboto', 'Arial', sans-serif;
                font-size: 12px;
                border: 1px solid #e0e4e8;
                border-radius: 8px;
            }
            QTableView::item {
                padding: 5px;
            }
            QTableView::item:hover {
                background-color: #f1f3f5;
            }
            QTableView::item:alternate {
                background-color: #f8f9fc;
            }
            QLabel {
                font-family: 'Roboto', 'Arial', sans-serif;
                font-size: 14px;
                background-color: #ffffff;
                border: 1px solid #e0e4e8;
                padding: 8px;
                border-radius: 5px;
            }
            QLineEdit, QDoubleSpinBox {
                border: 1px solid #e0e4e8;
                padding: 8px;
                border-radius: 5px;
                font-family: 'Roboto', 'Arial', sans-serif;
                font-size: 14px;
                background-color: #ffffff;
            }
            QLineEdit:focus, QDoubleSpinBox:focus {
                border: 2px solid #ff6b6b;
            }
            QProgressBar {
                border: 1px solid #e0e4e8;
                border-radius: 5px;
                text-align: center;
                font-family: 'Roboto', 'Arial', sans-serif;
                font-size: 12px;
                background-color: #f1f3f5;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ff6b6b, stop:1 #ee5253);
                border-radius: 3px;
            }
            QToolBar {
                background: #ffffff;
                border-bottom: 1px solid #e0e4e8;
                padding: 5px;
            }
            QToolBar QToolButton {
                background: transparent;
                padding: 5px;
                border: none;
            }
            QToolBar QToolButton:hover {
                background: #f1f3f5;
                border-radius: 5px;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
        """)

        # Main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(15)

        # Toolbar
        self.toolbar = QToolBar()
        self.addToolBar(self.toolbar)
        self.toolbar.setMovable(False)

        load_action = QAction(QIcon.fromTheme("document-open"), "Load Dataset", self)
        load_action.triggered.connect(self.load_dataset)
        self.toolbar.addAction(load_action)

        train_action = QAction(QIcon.fromTheme("system-run"), "Run Training", self)
        train_action.triggered.connect(self.run_training)
        self.toolbar.addAction(train_action)

        export_action = QAction(QIcon.fromTheme("document-save"), "Export Results", self)
        export_action.triggered.connect(self.export_results)
        self.toolbar.addAction(export_action)

        # Tabs
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        # Tab 1: Dataset
        self.dataset_tab = QWidget()
        self.dataset_layout = QVBoxLayout(self.dataset_tab)
        self.dataset_layout.setSpacing(10)
        self.tabs.addTab(self.dataset_tab, "Dataset")

        # Search bar
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search dataset...")
        self.search_bar.textChanged.connect(self.filter_table)
        self.dataset_layout.addWidget(self.search_bar)

        # Dataset table with filter model
        self.table_model = QStandardItemModel()
        self.proxy_model = QSortFilterProxyModel()
        self.proxy_model.setSourceModel(self.table_model)
        self.proxy_model.setFilterKeyColumn(-1)
        self.proxy_model.setFilterCaseSensitivity(Qt.CaseInsensitive)

        self.table = QTableView()
        self.table.setModel(self.proxy_model)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.dataset_layout.addWidget(self.table)

        # Tab 2: Training Results
        self.results_tab = QWidget()
        self.results_layout = QVBoxLayout(self.results_tab)
        self.results_layout.setSpacing(10)
        self.tabs.addTab(self.results_tab, "Training Results")

        # Metrics display
        self.metrics_label = QLabel("Metrics will appear here after training.")
        self.metrics_label.setFont(QFont("Roboto", 16, QFont.Bold))
        self.metrics_label.setAlignment(Qt.AlignCenter)
        self.results_layout.addWidget(self.metrics_label)

        # Remove the predictions display since test predictions are no longer generated
        self.results_layout.addStretch()

        # Tab 3: Visualization
        self.viz_tab = QWidget()
        self.viz_layout = QVBoxLayout(self.viz_tab)
        self.viz_layout.setSpacing(10)
        self.tabs.addTab(self.viz_tab, "Visualization")

        # Visualization label
        self.viz_label = QLabel("Accuracy vs Epochs Plot (Training and Validation)")
        self.viz_label.setFont(QFont("Roboto", 16, QFont.Bold))
        self.viz_label.setAlignment(Qt.AlignCenter)
        self.viz_layout.addWidget(self.viz_label)

        # Image display for the plot
        self.plot_label = QLabel()
        self.plot_label.setAlignment(Qt.AlignCenter)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.plot_label)
        self.scroll_area.setWidgetResizable(True)
        self.viz_layout.addWidget(self.scroll_area)

        # Tab 4: Manual Testing
        self.test_tab = QWidget()
        self.test_layout = QVBoxLayout(self.test_tab)
        self.test_layout.setSpacing(15)
        self.tabs.addTab(self.test_tab, "Manual Testing")

        # Form layout for input fields
        self.input_form = QFormLayout()
        self.input_form.setLabelAlignment(Qt.AlignRight)
        self.input_form.setSpacing(10)
        self.test_layout.addLayout(self.input_form)

        # Input fields for parameters
        self.temperature_input = QDoubleSpinBox()
        self.temperature_input.setRange(-50.0, 150.0)
        self.temperature_input.setValue(25.0)
        self.input_form.addRow("Temperature (°C):", self.temperature_input)

        self.humidity_input = QDoubleSpinBox()
        self.humidity_input.setRange(0.0, 100.0)
        self.humidity_input.setValue(50.0)
        self.input_form.addRow("Humidity (%):", self.humidity_input)

        self.gas_mq3_input = QDoubleSpinBox()
        self.gas_mq3_input.setRange(0.0, 1000.0)
        self.gas_mq3_input.setValue(0.0)
        self.input_form.addRow("Gas MQ3 (ppm):", self.gas_mq3_input)

        self.gas_mq135_input = QDoubleSpinBox()
        self.gas_mq135_input.setRange(0.0, 1000.0)
        self.gas_mq135_input.setValue(0.0)
        self.input_form.addRow("Gas MQ135 (ppm):", self.gas_mq135_input)

        # Predict button
        self.predict_button = QPushButton("Predict Fire Risk")
        self.predict_button.clicked.connect(self.run_prediction)
        self.predict_button.setMaximumWidth(200)
        self.predict_button.setStyleSheet("margin: 10px auto;")  # Center the button
        self.test_layout.addWidget(self.predict_button, alignment=Qt.AlignCenter)

        # Prediction result display
        self.test_result_label = QLabel("Prediction Result: Not available")
        self.test_result_label.setFont(QFont("Roboto", 16, QFont.Bold))
        self.test_result_label.setAlignment(Qt.AlignCenter)
        self.test_layout.addWidget(self.test_result_label)

        # Spacer to push content to the top
        self.test_layout.addStretch()

        # Progress bar and status
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.main_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("QLabel { color: #7f8c8d; font-style: italic; font-family: 'Roboto', 'Arial', sans-serif; font-size: 12px; border: none; }")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.status_label)

        # Next prediction
        self.next_pred_label = QLabel("Next fire risk prediction: Not available")
        self.next_pred_label.setFont(QFont("Roboto", 16, QFont.Bold))
        self.next_pred_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.next_pred_label)

        # Store results for export
        self.results = None

    def load_dataset(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_name:
            try:
                self.status_label.setText("Loading dataset...")
                df = pd.read_csv(file_name)
                self.display_dataset(df)
                self.status_label.setText(f"Dataset loaded: {file_name}")
            except Exception as e:
                self.status_label.setText(f"Error loading dataset: {str(e)}")
                QMessageBox.critical(self, "Error", f"Failed to load dataset: {str(e)}")

    def display_dataset(self, df):
        self.table_model.clear()
        self.table_model.setHorizontalHeaderLabels(df.columns)

        for i in range(df.shape[0]):
            row = []
            for j in range(df.shape[1]):
                item = QStandardItem(str(df.iloc[i, j]))
                item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                row.append(item)
            self.table_model.appendRow(row)

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def filter_table(self, text):
        self.proxy_model.setFilterWildcard(text)

    def run_training(self):
        try:
            self.status_label.setText("Starting training...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)

            result = subprocess.run(
                ["./target/release/fire_detection"],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                self.status_label.setText(f"Training failed: {result.stderr}")
                self.progress_bar.setVisible(False)
                QMessageBox.critical(self, "Error", f"Training failed: {result.stderr}")
                return

            # Load results from JSON
            with open("results.json", "r") as f:
                self.results = json.load(f)

            # Update metrics with training and validation accuracies (remove test accuracy)
            self.metrics_label.setText(
                f"Training Accuracy: {self.results['max_train_accuracy']:.2f}%\n"
                f"Validation Accuracy: {self.results['max_val_accuracy']:.2f}%"
            )

            # Load and display the accuracy vs epochs plot
            try:
                pixmap = QPixmap("hasil_prediksi.png")
                if pixmap.isNull():
                    raise FileNotFoundError("Could not load hasil_prediksi.png")
                self.plot_label.setPixmap(pixmap.scaled(800, 600, Qt.KeepAspectRatio))
            except FileNotFoundError as e:
                self.status_label.setText(f"Error loading plot: {str(e)}")
                QMessageBox.warning(self, "Warning", f"Plot file not found: {str(e)}")
                self.plot_label.setText("Plot not available")

            # Update next prediction
            self.next_pred_label.setText(
                f"Next fire risk prediction: {'Yes' if self.results['next_prediction'] == 1 else 'No'}"
            )

            self.status_label.setText("Training completed successfully")
            self.progress_bar.setVisible(False)

        except Exception as e:
            self.status_label.setText(f"Error during training: {str(e)}")
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Error during training: {str(e)}")

    def run_prediction(self):
        # Check if model and scaler files exist
        if not (os.path.exists("model.bin") and os.path.exists("scaler.bin")):
            QMessageBox.warning(self, "Warning", "Model not trained yet. Please run training first.")
            return

        try:
            self.status_label.setText("Running prediction...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)

            # Get input values
            temperature = self.temperature_input.value()
            humidity = self.humidity_input.value()
            gas_mq3 = self.gas_mq3_input.value()
            gas_mq135 = self.gas_mq135_input.value()

            # Run the prediction command
            result = subprocess.run(
                [
                    "./target/release/fire_detection",
                    "--predict",
                    str(temperature),
                    str(humidity),
                    str(gas_mq3),
                    str(gas_mq135)
                ],
                capture_output=True, text=True
            )
            print("stdout:", result.stdout)  # Debug print
            print("stderr:", result.stderr)  # Debug print

            if result.returncode != 0:
                self.status_label.setText(f"Prediction failed: {result.stderr}")
                self.progress_bar.setVisible(False)
                QMessageBox.critical(self, "Error", f"Prediction failed: {result.stderr}")
                return

            # Parse the prediction result
            prediction_result = json.loads(result.stdout)
            prediction = prediction_result["prediction"]
            probability = prediction_result["probability"]

            # Display the result
            result_text = (
                f"Prediction Result: {'Yes' if prediction == 1 else 'No'}\n"
                f"Probability: {probability:.2f}"
            )
            self.test_result_label.setText(result_text)

            self.status_label.setText("Prediction completed successfully")
            self.progress_bar.setVisible(False)

        except Exception as e:
            self.status_label.setText(f"Error during prediction: {str(e)}")
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Error during prediction: {str(e)}")

    def export_results(self):
        if self.results is None:
            QMessageBox.warning(self, "Warning", "No results available. Run training first.")
            return

        file_name, _ = QFileDialog.getSaveFileName(self, "Save Results", "", "CSV Files (*.csv)")
        if file_name:
            try:
                # Export training and validation accuracies and next prediction
                data = {
                    "Training Accuracy": [self.results["max_train_accuracy"]],
                    "Validation Accuracy": [self.results["max_val_accuracy"]],
                    "Next Prediction (1=Yes, 0=No)": [self.results["next_prediction"]]
                }
                df = pd.DataFrame(data)
                df.to_csv(file_name, index=False)
                self.status_label.setText(f"Results exported to {file_name}")
                QMessageBox.information(self, "Success", f"Results exported to {file_name}")
            except Exception as e:
                self.status_label.setText(f"Error exporting results: {str(e)}")
                QMessageBox.critical(self, "Error", f"Error exporting results: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = FireDetectionApp()
    window.show()
    sys.exit(app.exec_())