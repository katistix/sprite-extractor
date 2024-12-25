import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, 
                             QSlider, QHBoxLayout, QLineEdit, QMessageBox)
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
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color range for filtering colored objects
    lower_bound = np.array([0, 100, 100])  # Lower bound of color range (HSV)
    upper_bound = np.array([180, 255, 255])  # Upper bound of color range (HSV)
    
    # Include medium gray color detection (adjusting hue to account for grayish colors)
    lower_gray = np.array([0, 0, 50])  # Low lightness (gray)
    upper_gray = np.array([180, 50, 150])  # High lightness (gray)

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)

    # Combine the masks for color objects and gray lines
    combined_mask = cv2.bitwise_or(mask, mask_gray)

    # Apply Gaussian blur to smooth out noise
    blurred = cv2.GaussianBlur(combined_mask, (5, 5), 0)

    # Apply binary thresholding (you can adjust this threshold as necessary)
    _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

    # Morphological operation to remove small noise (small dots)
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours (dots)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]  # Keep only large contours

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
        self.preview_label.setFixedSize(800, 600)  # Larger preview size
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
        self.preview_label.setPixmap(pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
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
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color range for filtering colored objects
        lower_bound = np.array([0, 100, 100])
        upper_bound = np.array([180, 255, 255])
        
        # Include medium gray color detection
        lower_gray = np.array([0, 0, 50])
        upper_gray = np.array([180, 50, 150])

        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)

        # Combine the masks for color objects and gray lines
        combined_mask = cv2.bitwise_or(mask, mask_gray)

        # Apply Gaussian blur to smooth out noise
        blurred = cv2.GaussianBlur(combined_mask, (5, 5), 0)

        # Apply binary thresholding
        _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

        # Morphological operation to remove small noise
        kernel = np.ones((5, 5), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out small contours (dots)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]  # Keep only large contours

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
