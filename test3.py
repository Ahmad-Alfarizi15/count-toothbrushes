import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QListWidget
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer


class ProductCounterFiles(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Produk Counter (Multi File / Video / Webcam)")
        self.setGeometry(200, 200, 1000, 700)

        # Load model hasil training
        self.model = YOLO("best.pt")

        # Widget GUI
        self.label = QLabel("Pilih gambar / video / webcam untuk deteksi", self)
        self.label.setAlignment(Qt.AlignCenter)

        self.result_list = QListWidget(self)

        self.total_label = QLabel("", self)
        self.total_label.setAlignment(Qt.AlignCenter)

        self.button_img = QPushButton("Pilih File Gambar", self)
        self.button_img.clicked.connect(self.load_files)

        self.button_vid = QPushButton("Pilih File Video", self)
        self.button_vid.clicked.connect(self.load_video)

        self.button_cam = QPushButton("Pakai Webcam", self)
        self.button_cam.clicked.connect(self.load_webcam)

        self.button_stop = QPushButton("Stop", self)
        self.button_stop.clicked.connect(self.stop_video)

        self.button_resume = QPushButton("Resume", self)
        self.button_resume.clicked.connect(self.resume_video)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.result_list)
        layout.addWidget(self.total_label)
        layout.addWidget(self.button_img)
        layout.addWidget(self.button_vid)
        layout.addWidget(self.button_cam)
        layout.addWidget(self.button_stop)
        layout.addWidget(self.button_resume)

        self.setLayout(layout)

        # Timer untuk update video / webcam
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = None  # VideoCapture

    def draw_polygons_or_boxes(self, img, result):
        """Gambar polygon (jika segmentation), fallback ke box jika detection, sertakan label"""
        overlay = img.copy()

        # Segmentation model
        if result.masks is not None and len(result.masks.xy) > 0:
            for i, seg in enumerate(result.masks.xy):
                pts = seg.astype(int).reshape((-1, 1, 2))
                # garis polygon
                cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                # isi semi transparan
                cv2.fillPoly(overlay, [pts], color=(0, 255, 0))

                # Ambil class ID dari boxes (jika ada) atau default 0
                cls_id = int(result.boxes.cls[i]) if result.boxes is not None and len(result.boxes) > i else 0
                label = result.names.get(cls_id, "Object")
                x, y = pts[0][0]
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 0, 255), 5, cv2.LINE_AA)

            # gabung overlay
            img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

        # Detection model
        elif result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

                cls_id = int(box.cls[0])
                label = result.names.get(cls_id, "Object")
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (0, 0, 255), 3, cv2.LINE_AA)

        return img

    def load_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Pilih Gambar", "", "Image files (*.jpg *.png)"
        )
        if files:
            total_products = 0
            self.result_list.clear()

            for img_path in files:
                results = self.model.predict(source=img_path, conf=0.5, verbose=False)
                result = results[0]

                num_products = len(result.boxes)
                total_products += num_products

                img_name = os.path.basename(img_path)
                self.result_list.addItem(f"{img_name} → {num_products} produk")

            self.total_label.setText(f"TOTAL semua produk: {total_products}")

            # Tampilkan gambar terakhir dengan polygon/box
            last_img = files[-1]
            result = self.model.predict(source=last_img, conf=0.5, verbose=False)[0]
            img = cv2.imread(last_img)

            img = self.draw_polygons_or_boxes(img, result)
            self.show_image(img)

    def load_video(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Pilih Video", "", "Video files (*.mp4 *.avi *.mov)"
        )
        if file:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(file)
            self.timer.start(30)

    def load_webcam(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)  # default kamera laptop/webcam
        self.timer.start(30)

    def stop_video(self):
        if self.timer.isActive():
            self.timer.stop()

    def resume_video(self):
        if self.cap and not self.timer.isActive():
            self.timer.start(30)

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.timer.stop()
                self.cap.release()
                return

            results = self.model.predict(source=frame, conf=0.6, iou=0.4, verbose=False)
            result = results[0]

            num_products = len(result.boxes)
            self.total_label.setText(f"Produk terdeteksi (frame ini): {num_products}")

            frame = self.draw_polygons_or_boxes(frame, result)
            self.show_image(frame)

    def show_image(self, img):
        """Konversi OpenCV image ke QImage lalu tampilkan di QLabel"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Tampilkan dengan skala lebih kecil agar muat
        max_size = 800  # batas lebar/tinggi maksimum
        self.label.setPixmap(
            QPixmap.fromImage(qimg).scaled(max_size, max_size, Qt.KeepAspectRatio)
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ProductCounterFiles()
    window.show()
    sys.exit(app.exec_())
