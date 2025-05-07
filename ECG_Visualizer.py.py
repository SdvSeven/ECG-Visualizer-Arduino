import sys
import os
import csv
from datetime import datetime

import numpy as np
import serial
import serial.tools.list_ports
from scipy.fftpack import fft, ifft
from scipy.signal import find_peaks

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QFileDialog, QMessageBox,
    QCheckBox, QSpinBox
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPalette, QColor, QWheelEvent

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = (
    r"C:\\Users\\Администратор\\AppData\\Local\\Programs\\Python\\Python310"
    r"\\Lib\\site-packages\\PyQt5\\Qt5\\plugins"
)


class ECGVisualizer(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Визуализатор ЭКГ")
        self.resize(1200, 800)

        self.serial_port = None
        self.port_name = None
        self.data_buffer = []

        self.timer = QTimer()
        self.timer.timeout.connect(self.read_data)

        self.reconnect_timer = QTimer()
        self.reconnect_timer.timeout.connect(self.try_reconnect)
        self.reconnect_timer.start(5000)

        self.init_ui()
        self.auto_connect_serial()
        self.toggle_theme()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax_ecg = self.figure.add_subplot(311)
        self.ax_pulse = self.figure.add_subplot(312)
        self.ax_fft = self.figure.add_subplot(313)
        self.figure.subplots_adjust(hspace=0.5)

        self.init_plots()

        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.canvas)

        self.control_panel_container = QWidget()
        control_panel_layout = QVBoxLayout(self.control_panel_container)

        self.start_button = QPushButton("🟢ЗАПУСК")
        self.stop_button = QPushButton("🔴СТОП")
        self.pause_button = QPushButton("🟡ПАУЗА")
        self.save_data_button = QPushButton("💾Сохранить данные")

        button_size = (300, 100)
        font_size = 18

        for btn in [
            self.start_button, self.stop_button,
            self.pause_button, self.save_data_button
        ]:
            btn.setFixedSize(*button_size)
            btn.setStyleSheet(f"font-size: {font_size}px;")
            control_panel_layout.addWidget(btn)

        self.start_button.clicked.connect(self.start_acquisition)
        self.stop_button.clicked.connect(self.stop_acquisition)
        self.pause_button.clicked.connect(self.pause_acquisition)
        self.save_data_button.clicked.connect(self.save_ecg_data)

        freq_label = QLabel("Частота обновления: ")
        self.freq_selector = QSpinBox()
        self.freq_selector.setRange(30, 250)
        self.freq_selector.setValue(30)

        freq_layout = QHBoxLayout()
        freq_layout.addWidget(freq_label)
        freq_layout.addWidget(self.freq_selector)
        control_panel_layout.addLayout(freq_layout)

        self.dark_theme_checkbox = QCheckBox("                    Тёмная тема")
        self.dark_theme_checkbox.stateChanged.connect(self.toggle_theme)
        control_panel_layout.addWidget(self.dark_theme_checkbox)

        control_panel_layout.addStretch()
        main_layout.addLayout(plot_layout, stretch=3)
        main_layout.addWidget(self.control_panel_container, stretch=1)

    def init_plots(self):
        self.ax_ecg.clear()
        self.ax_pulse.clear()
        self.ax_fft.clear()

        self.ax_ecg.set_title("ЭКГ")
        self.ax_pulse.set_title("Частота сердечных сокращений (ударов в минуту)")
        self.ax_fft.set_title("Сглаженная ЭКГ (Фурье)")

        for ax in [self.ax_ecg, self.ax_pulse, self.ax_fft]:
            ax.grid(True, color='gray', alpha=0.3)

        self.canvas.draw()

    def toggle_theme(self):
        if self.dark_theme_checkbox.isChecked():
            palette = QPalette()
            palette.setColor(QPalette.Window, QColor(43, 43, 43))
            palette.setColor(QPalette.WindowText, Qt.white)
            self.setPalette(palette)
            self.figure.set_facecolor("#2b2b2b")
            for ax in self.figure.axes:
                ax.set_facecolor("#3c3f41")
                ax.tick_params(colors='white', labelcolor='white')
                ax.title.set_color('white')
                ax.grid(True, color='white', alpha=0.3)
            self.control_panel_container.setStyleSheet("""
                background-color: rgba(60, 63, 65, 0.5);
                border: 2px solid #5dade2;
                border-radius: 15px;
                padding: 10px;
            """)
        else:
            self.setPalette(QApplication.palette())
            self.figure.set_facecolor("#f0f9fa")
            for ax in self.figure.axes:
                ax.set_facecolor("white")
                ax.tick_params(colors='black', labelcolor='black')
                ax.title.set_color('black')
                ax.grid(True, color='gray', alpha=0.3)
            self.control_panel_container.setStyleSheet("""
                background-color: rgba(41, 128, 185, 0.1);
                border: 2px solid #2980b9;
                border-radius: 15px;
                padding: 10px;
            """)

        self.canvas.draw()

    def auto_connect_serial(self):
        ports = list(serial.tools.list_ports.comports())
        for port in ports:
            try:
                self.serial_port = serial.Serial(port.device, 9600)
                self.port_name = port.device
                return
            except serial.SerialException:
                continue
        self.serial_port = None
        self.port_name = None

    def try_reconnect(self):
        try:
            if self.serial_port and self.serial_port.is_open:
                return

            if self.port_name:
                try:
                    self.serial_port = serial.Serial(self.port_name, 9600)
                    print(f"[INFO] Переподключение к {self.port_name} успешно.")
                    return
                except serial.SerialException as e:
                    print(f"[ERROR] Не удалось переподключиться: {e}")
                    self.serial_port = None

            ports = list(serial.tools.list_ports.comports())
            for port in ports:
                try:
                    self.serial_port = serial.Serial(port.device, 9600)
                    self.port_name = port.device
                    print(f"[INFO] Устройство найдено и подключено: {self.port_name}")
                    return
                except serial.SerialException:
                    continue

            if not self.serial_port:
                print("[ERROR] Устройство не найдено.")
                QMessageBox.critical(
                    self,
                    "Устройство не найдено",
                    "Проверьте подключение ЭКГ-устройства к USB-порту."
                )
        except Exception as e:
            print(f"[CRITICAL] Ошибка в try_reconnect: {e}")
            QMessageBox.critical(self, "Критическая ошибка", str(e))

    def read_data(self):
        try:
            while self.serial_port and self.serial_port.in_waiting:
                try:
                    line = self.serial_port.readline().decode('utf-8').strip()
                    value = float(line)
                    self.data_buffer.append(value)
                    if len(self.data_buffer) > 1000:
                        self.data_buffer.pop(0)
                except Exception as e:
                    print(f"[WARNING] Ошибка при чтении данных: {e}")
                    continue
            self.update_plot()
        except Exception as e:
            print(f"[ERROR] Ошибка в read_data: {e}")
            QMessageBox.critical(
                self, "Ошибка чтения",
                f"Ошибка при чтении с устройства:\n{e}"
            )

    def update_plot(self):
        self.ax_ecg.clear()
        self.ax_pulse.clear()
        self.ax_fft.clear()

        data = np.array(self.data_buffer)
        fs = self.freq_selector.value()
        t = np.linspace(0, len(data) / fs, len(data))  # Время в секундах
        self.ax_ecg.plot(t, data, color="#0077cc", label="ECG")
        self.ax_ecg.set_xlabel("Время (с)")
        self.ax_ecg.set_title("ECG")
        self.ax_ecg.grid(True, color='gray', alpha=0.3)

        fft_data = fft(data)
        fft_data[200:] = 0
        smoothed_data = ifft(fft_data).real

        self.ax_fft.plot(t, smoothed_data, color="#00cccc", label="Smoothed ECG")
        self.ax_fft.set_xlabel("Время (с)")
        self.ax_fft.set_title("Smoothed ECG (Fourier)")
        self.ax_fft.grid(True, color='gray', alpha=0.3)

        peaks, _ = find_peaks(
            data, distance=30, height=np.mean(data) + np.std(data)
        )
        rr_intervals = np.diff(peaks)

        if len(rr_intervals) > 0:
            fs = self.freq_selector.value()
            rr_sec = rr_intervals / fs
            bpm = 60 / rr_sec
            avg_bpm = np.mean(bpm)
            self.ax_pulse.plot(
                peaks[1:], bpm, 'ro-', label=f"Pulse: {avg_bpm:.1f} bpm"
            )
            self.ax_pulse.set_ylim(0, 200)
        else:
            self.ax_pulse.plot([], [], 'ro-', label="Нет данных для пульса")

        self.ax_pulse.set_title("Heart Rate (уд/мин)")
        self.ax_pulse.grid(True, color='gray', alpha=0.3)

        for ax in [self.ax_ecg, self.ax_pulse, self.ax_fft]:
            ax.legend()

        self.canvas.draw()

    def start_acquisition(self):
        if self.serial_port:
            interval = int(1000 / self.freq_selector.value())
            self.timer.start(interval)

    def stop_acquisition(self):
        self.timer.stop()
        self.data_buffer.clear()
        self.init_plots()

    def pause_acquisition(self):
        self.timer.stop()

    def save_ecg_data(self):
        if not self.data_buffer:
            QMessageBox.warning(self, "Нет данных", "Нет данных для сохранения.")
            return

        data = np.array(self.data_buffer)
        peaks, _ = find_peaks(
            data, distance=30, height=np.mean(data) + np.std(data)
        )
        rr_intervals = np.diff(peaks)

        if len(rr_intervals) > 0:
            fs = self.freq_selector.value()
            rr_sec = rr_intervals / fs
            bpm = 60 / rr_sec
            min_bpm = np.min(bpm)
            avg_bpm = np.mean(bpm)
            max_bpm = np.max(bpm)
        else:
            min_bpm = avg_bpm = max_bpm = 0

        min_emg = np.min(data)
        avg_emg = np.mean(data)
        max_emg = np.max(data)

        file_name, _ = QFileDialog.getSaveFileName(
            self, "Сохранить данные ЭКГ", "", "CSV Files (*.csv)"
        )
        if file_name:
            with open(file_name, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    "Date", "Time", "Minimum pulse", "Average pulse",
                    "Maximum pulse", "Minimum EMG", "Average EMG", "Maximum EMG"
                ])
                current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([
                    current_datetime.split()[0],
                    current_datetime.split()[1],
                    round(min_bpm, 2),
                    round(avg_bpm, 2),
                    round(max_bpm, 2),
                    round(min_emg, 2),
                    round(avg_emg, 2),
                    round(max_emg, 2)
                ])
            QMessageBox.information(
                self, "Успех", "Данные ЭКГ успешно сохранены в CSV файл."
            )

    def wheelEvent(self, event: QWheelEvent):
        angle = event.angleDelta().y()
        factor = 1.1 if angle > 0 else 0.9

        for ax in [self.ax_ecg, self.ax_pulse, self.ax_fft]:
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            ax.set_xlim([x_min * factor, x_max * factor])
            ax.set_ylim([y_min * factor, y_max * factor])

        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ECGVisualizer()
    window.show()
    sys.exit(app.exec_())
