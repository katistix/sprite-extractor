import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, 
                             QSlider, QHBoxLayout, QLineEdit, QMessageBox, QSpinBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage

def cv_to_qt_image(cv_img):
    """Convert an OpenCV image to a QPixmap for display in QLabel."""
    height, width, channel = cv_img.shape
    bytes_per_line = 3 * width
    qt_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qt_img.rgbSwapped())

def extract_objects_preview(image_path, threshold_value):
    """Extract objects and provide a preview of the rectangles and the number of objects detected."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around contours
    preview_image = image.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(preview_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return preview_image, len(contours)

class ObjectExtractorApp(QWidget):
    def __init__(self):
        super().__init__()

        self.image_path = ""
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Upload button
        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)
        layout.addWidget(self.upload_button)

        # Label to display image path
        self.img_label = QLabel("No image uploaded")
        layout.addWidget(self.img_label)

        # Preview label
        self.preview_label = QLabel("Image Preview")
        self.preview_label.setFixedSize(400, 300)  # Adjust size as needed
        layout.addWidget(self.preview_label)

        # Threshold slider and entry
        threshold_layout = QHBoxLayout()
        self.threshold_label = QLabel("Threshold Value:")
        threshold_layout.addWidget(self.threshold_label)
        
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(255)
        self.threshold_slider.setValue(100)
        self.threshold_slider.valueChanged.connect(self.update_preview)
        threshold_layout.addWidget(self.threshold_slider)

        layout.addLayout(threshold_layout)

        # Object count label
        self.count_label = QLabel("Object Count: 0")
        layout.addWidget(self.count_label)

        # Extract/Export button
        self.extract_button = QPushButton("Extract/Export")
        self.extract_button.clicked.connect(self.extract_and_export)
        layout.addWidget(self.extract_button)

        self.setLayout(layout)
        self.setWindowTitle("Object Extractor with Real-Time Preview")

    def upload_image(self):
        options = QFileDialog.Options()
        self.image_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", 
                                                         "Image Files (*.jpg *.jpeg *.png)", options=options)
        if self.image_path:
            self.img_label.setText(self.image_path.split("/")[-1])
            self.update_preview()

    def update_preview(self):
        if not self.image_path:
            return
        
        threshold_value = self.threshold_slider.value()
        preview_image, object_count = extract_objects_preview(self.image_path, threshold_value)

        # Convert to QPixmap and display
        pixmap = cv_to_qt_image(preview_image)
        self.preview_label.setPixmap(pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio))
        
        # Update object count
        self.count_label.setText(f"Object Count: {object_count}")

    def extract_and_export(self):
        if not self.image_path:
            QMessageBox.critical(self, "Error", "Please upload an image first.")
            return
        
        output_folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        
        if not output_folder:
            QMessageBox.critical(self, "Error", "Please select an output folder.")
            return
        
        threshold_value = self.threshold_slider.value()
        self.extract_objects(self.image_path, threshold_value, output_folder)
        QMessageBox.information(self, "Success", "Objects extracted and saved successfully!")

    def extract_objects(self, image_path, threshold_value, output_folder):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract and save each object
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)

            mask = np.zeros(image.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            rect = (x, y, w, h)
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            extracted_object_gc = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
            extracted_object_gc[:, :, :3] = image * mask2[:, :, np.newaxis]
            extracted_object_gc[:, :, 3] = mask2 * 255

            cropped_object_gc = extracted_object_gc[y:y + h, x:x + w]
            output_path = f"{output_folder}/extracted_object_{i}.png"
            cv2.imwrite(output_path, cropped_object_gc)

def main():
    app = QApplication(sys.argv)
    ex = ObjectExtractorApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
