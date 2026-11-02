# save as video_to_images_gui.py
import sys, os, cv2, math, time
from dataclasses import dataclass
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QFileDialog,
    QHBoxLayout, QVBoxLayout, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QProgressBar, QGroupBox, QMessageBox
)

@dataclass
class ExtractConfig:
    video_path: str
    output_dir: str
    mode: str                   # "every_frame" | "every_n_frames" | "every_x_seconds"
    n_frames: int               # used if mode == every_n_frames
    x_seconds: float            # used if mode == every_x_seconds
    start_sec: float
    end_sec: float              # <= 0 means till end
    img_format: str             # "jpg" | "png" | "bmp"
    jpg_quality: int
    png_compression: int
    include_timestamp: bool
    resize_percent: int         # 100 = original
    overwrite: bool

class ExtractWorker(QObject):
    progress = pyqtSignal(int)               # 0..100
    status = pyqtSignal(str)
    finished = pyqtSignal(int, int, float)   # saved_count, total_checked, elapsed_s
    error = pyqtSignal(str)

    def __init__(self, cfg: ExtractConfig):
        super().__init__()
        self.cfg = cfg
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        t0 = time.time()
        saved = 0
        checked = 0

        if not os.path.isfile(self.cfg.video_path):
            self.error.emit("File video tidak ditemukan.")
            return
        os.makedirs(self.cfg.output_dir, exist_ok=True)

        cap = cv2.VideoCapture(self.cfg.video_path)
        if not cap.isOpened():
            self.error.emit("Gagal membuka video (format/codec tidak didukung).")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = total_frames / fps if fps > 0 else 0.0

        start_ms = max(0.0, self.cfg.start_sec) * 1000.0
        end_ms = (self.cfg.end_sec * 1000.0) if (self.cfg.end_sec > 0) else (duration * 1000.0)
        end_ms = max(end_ms, start_ms)

        # seek ke start
        cap.set(cv2.CAP_PROP_POS_MSEC, start_ms)

        # Interval logika
        save_every_n = max(1, self.cfg.n_frames)
        save_every_sec = max(1e-6, self.cfg.x_seconds)  # hindari nol

        next_save_time_ms = start_ms  # untuk mode every_x_seconds
        next_save_frame_idx = None
        if fps > 0:
            # frame index di sekitar start_ms
            start_frame_idx = int(round((start_ms / 1000.0) * fps))
            next_save_frame_idx = start_frame_idx

        encode_params = []
        if self.cfg.img_format.lower() == "jpg":
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(self.cfg.jpg_quality)]
        elif self.cfg.img_format.lower() == "png":
            # 0 (no compression) .. 9 (max compression)
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, int(self.cfg.png_compression)]

        # helper nama file
        def make_filename(idx: int, pos_ms: float):
            if self.cfg.include_timestamp:
                ts = pos_ms / 1000.0
                return f"frame_{idx:06d}_t{ts:.3f}.{self.cfg.img_format}"
            else:
                return f"frame_{idx:06d}.{self.cfg.img_format}"

        # resizing factor
        scale = max(1, self.cfg.resize_percent) / 100.0

        # estimasi untuk progress
        span_ms = max(1.0, end_ms - start_ms)

        while True:
            if self._cancel:
                self.status.emit("Dibatalkan pengguna.")
                break

            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_ms >= end_ms:
                break

            ret, frame = cap.read()
            if not ret:
                # Mungkin mencapai akhir file
                break

            checked += 1
            curr_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            # hitung frame index saat ini (aproksimasi)
            if fps > 0:
                curr_frame_idx = int(round((curr_ms / 1000.0) * fps))
            else:
                curr_frame_idx = checked  # fallback

            # Tentukan apakah disimpan
            save_this = False
            if self.cfg.mode == "every_frame":
                save_this = True
            elif self.cfg.mode == "every_n_frames":
                if fps > 0:
                    # Simpan bila melewati frame target berikutnya
                    if curr_frame_idx >= (next_save_frame_idx or 0):
                        save_this = True
                        next_save_frame_idx = curr_frame_idx + save_every_n
                else:
                    # jika fps tak diketahui, fallback: simpan tiap iterasi ke-n
                    if checked % save_every_n == 0:
                        save_this = True
            elif self.cfg.mode == "every_x_seconds":
                if curr_ms + 0.5 >= next_save_time_ms:
                    save_this = True
                    next_save_time_ms += (save_every_sec * 1000.0)

            if save_this:
                # optional resize
                if scale != 1.0:
                    new_w = max(1, int(round(width * scale)))
                    new_h = max(1, int(round(height * scale)))
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                fname = make_filename(curr_frame_idx, curr_ms)
                out_path = os.path.join(self.cfg.output_dir, fname)

                if (not self.cfg.overwrite) and os.path.exists(out_path):
                    # cari nama unik
                    base, ext = os.path.splitext(out_path)
                    k = 1
                    while os.path.exists(f"{base}_{k}{ext}"):
                        k += 1
                    out_path = f"{base}_{k}{ext}"

                ok = cv2.imwrite(out_path, frame, encode_params)
                if not ok:
                    self.status.emit(f"Gagal menyimpan: {os.path.basename(out_path)}")
                else:
                    saved += 1
                    if saved % 25 == 0:
                        self.status.emit(f"Tersimpan {saved} gambar...")

            # progress
            prog = int(((curr_ms - start_ms) / span_ms) * 100.0)
            self.progress.emit(min(100, max(0, prog)))

        cap.release()
        elapsed = time.time() - t0
        if not self._cancel:
            self.progress.emit(100)
            self.status.emit(f"Selesai. Tersimpan {saved} gambar.")
        self.finished.emit(saved, checked, elapsed)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video → Images (OpenCV) | PyQt5")
        self.setMinimumWidth(700)

        # === Path group ===
        self.le_video = QLineEdit()
        btn_browse_video = QPushButton("Pilih Video")
        btn_browse_video.clicked.connect(self.browse_video)

        self.le_out = QLineEdit()
        btn_browse_out = QPushButton("Pilih Folder Output")
        btn_browse_out.clicked.connect(self.browse_out)

        g_path = QGroupBox("Sumber & Tujuan")
        l1 = QHBoxLayout(); l1.addWidget(QLabel("Video:")); l1.addWidget(self.le_video); l1.addWidget(btn_browse_video)
        l2 = QHBoxLayout(); l2.addWidget(QLabel("Output:")); l2.addWidget(self.le_out); l2.addWidget(btn_browse_out)
        vpath = QVBoxLayout(); vpath.addLayout(l1); vpath.addLayout(l2)
        g_path.setLayout(vpath)

        # === Info video ===
        self.lbl_info = QLabel("Info video: -")
        self.lbl_info.setWordWrap(True)

        # === Mode group ===
        self.cb_mode = QComboBox()
        self.cb_mode.addItems(["Setiap Frame", "Setiap N Frame", "Setiap X Detik"])
        self.cb_mode.currentIndexChanged.connect(self.on_mode_changed)

        self.spin_n = QSpinBox(); self.spin_n.setRange(1, 100000); self.spin_n.setValue(30)
        self.dspin_x = QDoubleSpinBox(); self.dspin_x.setRange(0.001, 3600.0); self.dspin_x.setDecimals(3); self.dspin_x.setValue(1.0)

        g_mode = QGroupBox("Mode Ekstraksi")
        hmode = QHBoxLayout()
        hmode.addWidget(QLabel("Mode:")); hmode.addWidget(self.cb_mode)
        hmode.addSpacing(16)
        hmode.addWidget(QLabel("N frame:")); hmode.addWidget(self.spin_n)
        hmode.addSpacing(16)
        hmode.addWidget(QLabel("X detik:")); hmode.addWidget(self.dspin_x)
        g_mode.setLayout(hmode)
        self.on_mode_changed()  # set enable/disable

        # === Rentang waktu ===
        self.dspin_start = QDoubleSpinBox(); self.dspin_start.setRange(0.0, 1e8); self.dspin_start.setDecimals(3); self.dspin_start.setValue(0.0)
        self.dspin_end = QDoubleSpinBox(); self.dspin_end.setRange(0.0, 1e8); self.dspin_end.setDecimals(3); self.dspin_end.setValue(0.0)
        hint = QLabel("(End 0 = sampai akhir)")
        g_range = QGroupBox("Rentang Waktu (detik)")
        hr = QHBoxLayout()
        hr.addWidget(QLabel("Start:")); hr.addWidget(self.dspin_start)
        hr.addSpacing(16)
        hr.addWidget(QLabel("End:")); hr.addWidget(self.dspin_end)
        hr.addSpacing(16); hr.addWidget(hint)
        g_range.setLayout(hr)

        # === Output options ===
        self.cb_format = QComboBox(); self.cb_format.addItems(["jpg", "png", "bmp"])
        self.cb_format.currentIndexChanged.connect(self.on_fmt_changed)
        self.spin_jpg_q = QSpinBox(); self.spin_jpg_q.setRange(1, 100); self.spin_jpg_q.setValue(95)
        self.spin_png_c = QSpinBox(); self.spin_png_c.setRange(0, 9); self.spin_png_c.setValue(3)
        self.chk_ts = QCheckBox("Sertakan timestamp pada nama file")
        self.chk_ow = QCheckBox("Overwrite bila file sudah ada"); self.chk_ow.setChecked(False)
        self.spin_resize = QSpinBox(); self.spin_resize.setRange(1, 400); self.spin_resize.setValue(100)

        g_out = QGroupBox("Opsi Output")
        ho = QHBoxLayout()
        ho.addWidget(QLabel("Format:")); ho.addWidget(self.cb_format)
        ho.addSpacing(12)
        ho.addWidget(QLabel("JPG quality:")); ho.addWidget(self.spin_jpg_q)
        ho.addSpacing(12)
        ho.addWidget(QLabel("PNG compression:")); ho.addWidget(self.spin_png_c)
        ho.addSpacing(12)
        ho.addWidget(QLabel("Resize %:")); ho.addWidget(self.spin_resize)
        ho.addSpacing(12)
        ho.addWidget(self.chk_ts)
        ho.addSpacing(12)
        ho.addWidget(self.chk_ow)
        g_out.setLayout(ho)
        self.on_fmt_changed()

        # === Controls ===
        self.btn_probe = QPushButton("Cek Info Video")
        self.btn_probe.clicked.connect(self.probe_video)
        self.btn_start = QPushButton("Mulai Ekstraksi")
        self.btn_start.clicked.connect(self.start_extract)
        self.btn_cancel = QPushButton("Batal"); self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_extract)

        hctrl = QHBoxLayout()
        hctrl.addWidget(self.btn_probe)
        hctrl.addStretch(1)
        hctrl.addWidget(self.btn_start)
        hctrl.addWidget(self.btn_cancel)

        self.pbar = QProgressBar(); self.pbar.setRange(0, 100)
        self.lbl_status = QLabel("Status: idle")

        # === Layout utama ===
        root = QVBoxLayout()
        root.addWidget(g_path)
        root.addWidget(self.lbl_info)
        root.addWidget(g_mode)
        root.addWidget(g_range)
        root.addWidget(g_out)
        root.addLayout(hctrl)
        root.addWidget(self.pbar)
        root.addWidget(self.lbl_status)
        self.setLayout(root)

        self.thread = None
        self.worker = None

    def on_mode_changed(self):
        mode = self.cb_mode.currentText()
        self.spin_n.setEnabled(mode == "Setiap N Frame")
        self.dspin_x.setEnabled(mode == "Setiap X Detik")

    def on_fmt_changed(self):
        fmt = self.cb_format.currentText().lower()
        self.spin_jpg_q.setEnabled(fmt == "jpg")
        self.spin_png_c.setEnabled(fmt == "png")

    def browse_video(self):
        fp, _ = QFileDialog.getOpenFileName(self, "Pilih Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov *.m4v *.ts *.mjpeg *.mpg *.wmv);;All Files (*)")
        if fp:
            self.le_video.setText(fp)

    def browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Pilih Folder Output", "")
        if d:
            self.le_out.setText(d)

    def probe_video(self):
        fp = self.le_video.text().strip()
        if not fp or not os.path.isfile(fp):
            QMessageBox.warning(self, "Perhatian", "Pilih file video yang valid.")
            return

        cap = cv2.VideoCapture(fp)
        if not cap.isOpened():
            self.lbl_info.setText("Info video: Gagal membuka video.")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = n / fps if fps > 0 else 0.0
        self.lbl_info.setText(f"Info video: {w}x{h}, {fps:.3f} FPS, {n} frames, durasi ~{duration:.3f} s")
        cap.release()

        # Isi default end detik bila 0
        if self.dspin_end.value() == 0.0 and duration > 0:
            self.dspin_end.setValue(round(duration, 3))

    def build_config(self) -> ExtractConfig:
        mode_idx = self.cb_mode.currentIndex()
        if mode_idx == 0: mode_key = "every_frame"
        elif mode_idx == 1: mode_key = "every_n_frames"
        else: mode_key = "every_x_seconds"

        return ExtractConfig(
            video_path=self.le_video.text().strip(),
            output_dir=self.le_out.text().strip() or os.path.join(os.path.dirname(self.le_video.text().strip() or "."), "frames"),
            mode=mode_key,
            n_frames=self.spin_n.value(),
            x_seconds=float(self.dspin_x.value()),
            start_sec=float(self.dspin_start.value()),
            end_sec=float(self.dspin_end.value()),
            img_format=self.cb_format.currentText().lower(),
            jpg_quality=self.spin_jpg_q.value(),
            png_compression=self.spin_png_c.value(),
            include_timestamp=self.chk_ts.isChecked(),
            resize_percent=self.spin_resize.value(),
            overwrite=self.chk_ow.isChecked()
        )

    def start_extract(self):
        cfg = self.build_config()

        if not cfg.video_path or not os.path.isfile(cfg.video_path):
            QMessageBox.warning(self, "Perhatian", "Pilih file video yang valid.")
            return

        if not cfg.output_dir:
            QMessageBox.warning(self, "Perhatian", "Pilih folder output (atau biarkan default).")
            return

        os.makedirs(cfg.output_dir, exist_ok=True)

        self.thread = QThread()
        self.worker = ExtractWorker(cfg)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.pbar.setValue)
        self.worker.status.connect(self.set_status)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.set_status("Memulai ekstraksi...")
        self.pbar.setValue(0)

        self.thread.start()

    def cancel_extract(self):
        if self.worker:
            self.worker.cancel()
            self.set_status("Mencoba membatalkan...")

    def set_status(self, txt: str):
        self.lbl_status.setText(f"Status: {txt}")

    def on_error(self, msg: str):
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.set_status(f"Error: {msg}")
        QMessageBox.critical(self, "Error", msg)

    def on_finished(self, saved: int, checked: int, elapsed: float):
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.set_status(f"Selesai. Disimpan {saved} gambar dari {checked} frame/cek dalam {elapsed:.2f} s.")
        QMessageBox.information(self, "Selesai", f"Disimpan {saved} gambar.\nFolder: {self.build_config().output_dir}")

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
