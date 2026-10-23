#!/usr/bin/env python3
"""Monocular depth estimation viewer for ROS 2 bag files (transformer based).

Run:  ros2 run mono_depth_viewer viewer [bag_folder_or_db3]
"""
import os
import sys
import threading
import time

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

# cv2 points Qt at its own bundled plugins, which clash with PyQt5's; use PyQt5's instead.
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ.pop("QT_PLUGIN_PATH", None)
QtCore.QCoreApplication.setLibraryPaths(
    [QtCore.QLibraryInfo.location(QtCore.QLibraryInfo.PluginsPath)])

from mono_depth_viewer.bag_reader import BagReader
from mono_depth_viewer.depth_model import COLORMAPS, MODELS, DepthEstimator

DEFAULT_TOPIC = "/camera/image_raw"


def to_pixmap(rgb):
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    return QtGui.QPixmap.fromImage(QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy())


class ImageView(QtWidgets.QLabel):
    def __init__(self, title):
        super().__init__(title)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.setStyleSheet("background:#111;color:#888;")
        self._pm = None

    def show_rgb(self, rgb):
        self._pm = to_pixmap(rgb)
        self._rescale()

    def _rescale(self):
        if self._pm:
            self.setPixmap(self._pm.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rescale()


class Worker(QtCore.QThread):
    """Loads the model and runs inference on the most recent requested frame (older ones are dropped)."""
    loaded = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)
    result = QtCore.pyqtSignal(int, object, object, float)   # index, rgb, depth, ms
    progress = QtCore.pyqtSignal(int, int)
    exported = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.cv = threading.Condition()
        self.jobs = []          # ("load", name) | ("frame", k, rgb, size) | ("export", fn)
        self.est = None
        self.cancel_export = False
        self._quit = False

    def submit(self, job):
        with self.cv:
            if job[0] == "frame":
                self.jobs = [j for j in self.jobs if j[0] != "frame"]
            self.jobs.append(job)
            self.cv.notify()

    def stop(self):
        with self.cv:
            self._quit = True
            self.cancel_export = True
            self.cv.notify()
        self.wait()

    def run(self):
        while True:
            with self.cv:
                while not self.jobs and not self._quit:
                    self.cv.wait()
                if self._quit:
                    return
                job = self.jobs.pop(0)
            try:
                if job[0] == "load":
                    self.est = None
                    self.est = DepthEstimator(job[1])
                    self.loaded.emit(f"{job[1]} on {self.est.device}")
                elif job[0] == "frame" and self.est:
                    _, k, rgb, size = job
                    t = time.perf_counter()
                    depth = self.est.predict(rgb, size)
                    self.result.emit(k, rgb, depth, (time.perf_counter() - t) * 1000)
                elif job[0] == "export" and self.est:
                    self.exported.emit(job[1](self))
            except Exception as e:  # keep the thread alive; report in the UI
                self.failed.emit(f"{type(e).__name__}: {e}")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Monocular Depth from ROS Bag")
        self.resize(1400, 760)
        self.reader = None
        self.k = 0
        self.playing = False
        self.busy = False
        self.pending = None
        self.last = None   # (rgb, depth)
        self.model_ready = False

        self.worker = Worker()
        self.worker.loaded.connect(self.on_loaded)
        self.worker.failed.connect(self.on_failed)
        self.worker.result.connect(self.on_result)
        self.worker.progress.connect(self.on_progress)
        self.worker.exported.connect(lambda p: self.status(f"Saved {p}"))
        self.worker.start()

        self._build_ui()
        self.load_model()

    # ---------- UI ----------
    def _build_ui(self):
        self.rgb_view, self.depth_view = ImageView("Open a bag to begin"), ImageView("Depth")
        views = QtWidgets.QHBoxLayout()
        views.addWidget(self.rgb_view)
        views.addWidget(self.depth_view)

        self.btn_open = QtWidgets.QPushButton("Open bag folder…")
        self.btn_open_file = QtWidgets.QPushButton("Open .db3…")
        self.cb_topic = QtWidgets.QComboBox()
        self.cb_topic.setMinimumWidth(200)
        self.cb_model = QtWidgets.QComboBox()
        self.cb_model.addItems(MODELS)
        self.cb_size = QtWidgets.QComboBox()
        self.cb_size.addItems(["252", "364", "518", "672"])
        self.cb_size.setCurrentText("364")
        self.cb_size.setToolTip("Short-side resolution fed to the network (smaller = faster)")
        self.cb_cmap = QtWidgets.QComboBox()
        self.cb_cmap.addItems(COLORMAPS)
        top = QtWidgets.QHBoxLayout()
        for w in (self.btn_open, self.btn_open_file, QtWidgets.QLabel("Topic"), self.cb_topic,
                  QtWidgets.QLabel("Model"), self.cb_model, QtWidgets.QLabel("Input"), self.cb_size,
                  QtWidgets.QLabel("Colormap"), self.cb_cmap):
            top.addWidget(w)
        top.addStretch()

        self.btn_play = QtWidgets.QPushButton("▶ Play")
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setEnabled(False)
        self.sp_stride = QtWidgets.QSpinBox()
        self.sp_stride.setRange(1, 100)
        self.sp_stride.setPrefix("every ")
        self.sp_stride.setSuffix(" frame(s)")
        self.btn_save = QtWidgets.QPushButton("Save frame")
        self.btn_export = QtWidgets.QPushButton("Export video…")
        bot = QtWidgets.QHBoxLayout()
        for w in (self.btn_play, self.slider, self.sp_stride, self.btn_save, self.btn_export):
            bot.addWidget(w)
        bot.setStretch(1, 1)

        lay = QtWidgets.QVBoxLayout()
        lay.addLayout(top)
        lay.addLayout(views, 1)
        lay.addLayout(bot)
        c = QtWidgets.QWidget()
        c.setLayout(lay)
        self.setCentralWidget(c)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)

        self.btn_open.clicked.connect(self.open_folder)
        self.btn_open_file.clicked.connect(self.open_file)
        self.cb_topic.activated.connect(lambda _: self.select_topic(self.cb_topic.currentText()))
        self.cb_model.activated.connect(lambda _: self.load_model())
        self.cb_size.activated.connect(lambda _: self.request(self.k))
        self.cb_cmap.activated.connect(lambda _: self.redraw())
        self.btn_play.clicked.connect(self.toggle_play)
        self.slider.valueChanged.connect(self.on_slider)
        self.btn_save.clicked.connect(self.save_frame)
        self.btn_export.clicked.connect(self.export_video)

    def status(self, msg):
        self.statusBar().showMessage(msg)

    # ---------- model ----------
    def load_model(self):
        self.model_ready = False
        self.status(f"Loading {self.cb_model.currentText()} (first run downloads weights)…")
        self.worker.submit(("load", self.cb_model.currentText()))

    def on_loaded(self, msg):
        self.model_ready = True
        self.status(f"Model ready: {msg}")
        if self.reader:
            self.request(self.k)

    def on_failed(self, msg):
        self.busy = False
        self.playing = False
        self.btn_play.setText("▶ Play")
        self.status(msg)
        QtWidgets.QMessageBox.warning(self, "Error", msg)

    # ---------- bag ----------
    def open_folder(self):
        p = QtWidgets.QFileDialog.getExistingDirectory(self, "Select rosbag2 folder", os.path.expanduser("~"))
        if p:
            self.open_bag(p)

    def open_file(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select .db3", os.path.expanduser("~"), "rosbag2 (*.db3)")
        if p:
            self.open_bag(p)

    def open_bag(self, path):
        try:
            reader = BagReader(path)
            if not reader.topics:
                raise RuntimeError("No sensor_msgs/Image or CompressedImage topics in this bag")
        except Exception as e:
            self.on_failed(str(e))
            return
        if self.reader:
            self.reader.close()
        self.reader = reader
        self.cb_topic.clear()
        self.cb_topic.addItems(sorted(reader.topics))
        if DEFAULT_TOPIC in reader.topics:
            self.cb_topic.setCurrentText(DEFAULT_TOPIC)
        self.select_topic(self.cb_topic.currentText())

    def select_topic(self, topic):
        n = self.reader.select_topic(topic)
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(n - 1, 0))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.slider.setEnabled(n > 0)
        self.k = 0
        self.status(f"{topic}: {n} frames")
        if n:
            self.request(0)

    # ---------- inference / playback ----------
    def request(self, k):
        if not self.reader or not len(self.reader):
            return
        self.k = k = max(0, min(k, len(self.reader) - 1))
        if not self.model_ready or self.busy:
            self.pending = k
            if not self.model_ready:
                self.rgb_view.show_rgb(self.reader.frame(k))
            return
        self.busy = True
        self.pending = None
        self.worker.submit(("frame", k, self.reader.frame(k), int(self.cb_size.currentText())))

    def on_result(self, k, rgb, depth, ms):
        self.busy = False
        self.last = (rgb, depth)
        self.redraw()
        t = self.reader.timestamp(k) - self.reader.timestamp(0)
        self.status(f"frame {k + 1}/{len(self.reader)}  t={t:.2f}s  inference {ms:.0f} ms ({1000 / ms:.1f} FPS)")
        self.slider.blockSignals(True)
        self.slider.setValue(k)
        self.slider.blockSignals(False)
        if self.pending is not None:
            self.request(self.pending)
        elif self.playing:
            nxt = k + self.sp_stride.value()
            if nxt >= len(self.reader):
                self.toggle_play()
            else:
                self.request(nxt)

    def redraw(self):
        if self.last:
            rgb, depth = self.last
            self.rgb_view.show_rgb(rgb)
            self.depth_view.show_rgb(self.worker.est.colorize(depth, self.cb_cmap.currentText()))

    def on_slider(self, v):
        self.request(v)

    def toggle_play(self):
        self.playing = not self.playing
        self.btn_play.setText("⏸ Pause" if self.playing else "▶ Play")
        if self.playing:
            if self.worker.est:
                self.worker.est.reset_range()
            self.request(self.k + (self.sp_stride.value() if self.k + 1 < len(self.reader or []) else 0))

    # ---------- output ----------
    def save_frame(self):
        if not self.last:
            return
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save depth frame", f"depth_{self.k:05d}.png", "PNG (*.png)")
        if not p:
            return
        rgb, depth = self.last
        color = self.worker.est.colorize(depth, self.cb_cmap.currentText(), smooth=0)
        cv2.imwrite(p, cv2.cvtColor(np.hstack([rgb, color]), cv2.COLOR_RGB2BGR))
        # raw relative depth for downstream use
        np.save(os.path.splitext(p)[0] + ".npy", depth.astype(np.float32))
        self.status(f"Saved {p} (+ .npy raw relative depth)")

    def export_video(self):
        if not (self.reader and self.model_ready) or self.busy:
            self.status("Wait for the current frame to finish, then try again")
            return
        if self.playing:
            self.toggle_play()
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export video", "depth.mp4", "MP4 (*.mp4)")
        if not p:
            return
        stride, size, cmap = self.sp_stride.value(), int(self.cb_size.currentText()), self.cb_cmap.currentText()
        reader, n = self.reader, len(self.reader)
        dur = reader.timestamp(n - 1) - reader.timestamp(0)
        fps = max(1.0, (n / dur if dur > 0 else 10.0) / stride)

        def job(w):
            est, out = w.est, None
            est.reset_range()
            w.cancel_export = False
            idx = list(range(0, n, stride))
            for i, k in enumerate(idx):
                if w.cancel_export:
                    break
                rgb = reader.frame(k)   # sqlite read-only connection, used only while GUI is idle
                color = est.colorize(est.predict(rgb, size), cmap)
                frame = cv2.cvtColor(np.hstack([rgb, color]), cv2.COLOR_RGB2BGR)
                if out is None:
                    out = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
                out.write(frame)
                w.progress.emit(i + 1, len(idx))
            if out:
                out.release()
            w.progress.emit(0, 0)
            return p

        self.set_controls_enabled(False)
        self.worker.submit(("export", job))

    def on_progress(self, i, total):
        if total == 0:
            self.progress.hide()
            self.set_controls_enabled(True)
            return
        self.progress.show()
        self.progress.setMaximum(total)
        self.progress.setValue(i)
        self.status(f"Exporting {i}/{total}")

    def set_controls_enabled(self, on):
        for w in (self.slider, self.btn_play, self.btn_save, self.btn_export, self.btn_open, self.btn_open_file,
                  self.cb_topic, self.cb_model):
            w.setEnabled(on)

    def closeEvent(self, e):
        self.worker.stop()
        super().closeEvent(e)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    if len(sys.argv) > 1:
        QtCore.QTimer.singleShot(0, lambda: w.open_bag(sys.argv[1]))
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
