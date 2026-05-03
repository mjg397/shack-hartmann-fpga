from __future__ import annotations

import os
import sys
from collections.abc import MutableMapping
from typing import cast
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Ensure shwfs_utils is in path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import shwfs_utils
from fpga_gui_service import (
    DEFAULT_BIND_IP,
    DEFAULT_BIND_PORT,
    FpgaControlService,
    RunResult,
    NUM_VALID_SUBAPERTURES,
)

# ===========================================================================
# STYLING CONSTANTS (PREMIUM DARK THEME)
# ===========================================================================
BG_COLOR = "#0B0E14"
SURFACE_COLOR = "#1A202C"
ACCENT_COLOR = "#00D1FF"  # Cyan
SUCCESS_COLOR = "#00F59B" # Green
ERROR_COLOR = "#FF4B4B"   # Red
TEXT_COLOR = "#E0E0E0"
SECONDARY_TEXT = "#A0AEC0"
BORDER_COLOR = "#2D3748"

# ===========================================================================
# CUSTOM WIDGETS
# ===========================================================================

class InfoCard(QtWidgets.QFrame):
    """A premium-styled information card."""
    def __init__(self, title: str, value: str = "n/a", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setStyleSheet(f"""
            InfoCard {{
                background-color: {SURFACE_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 12px;
                padding: 10px;
            }}
        """)
        
        layout = QtWidgets.QVBoxLayout(self)
        self.title_label = QtWidgets.QLabel(title.upper())
        self.title_label.setStyleSheet(f"color: {SECONDARY_TEXT}; font-size: 10px; font-weight: bold; letter-spacing: 1px;")
        
        self.value_label = QtWidgets.QLabel(value)
        self.value_label.setStyleSheet(f"color: {ACCENT_COLOR}; font-size: 18px; font-weight: bold;")
        self.value_label.setWordWrap(True)
        
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addStretch()

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)

class PlotCanvas(FigureCanvasQTAgg):
    """Enhanced Matplotlib canvas with premium styling."""
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        self.figure = Figure(figsize=(12, 10), facecolor=BG_COLOR)
        super().__init__(self.figure)
        self.setParent(parent)
        plt.style.use("dark_background")

    def update_result(self, result: RunResult) -> None:
        self.figure.clear()
        # 2x2 Grid for standard telemetry
        axes = self.figure.subplots(2, 2)
        self.figure.subplots_adjust(hspace=0.3, wspace=0.2)

        # 1. Detector Image
        img = np.asarray(result.quantized_image)
        if img.ndim == 1:
            side = int(np.sqrt(img.size))
            img = img.reshape(side, side)
        
        ax0 = axes[0, 0]
        ax0.imshow(img, cmap="magma", origin="lower")
        ax0.set_title("DETECTOR FRAME", color=TEXT_COLOR, fontsize=10, fontweight="bold")
        ax0.axis("off")

        # 2. Slopes Field (Quiver)
        ax1 = axes[0, 1]
        if result.slopes_xy_grid is not None:
            # We use a subset for visualization if it's too dense, but 16x16 is fine
            grid = result.slopes_xy_grid
            sx = grid[..., 0]
            sy = grid[..., 1]
            ax1.imshow(img, cmap="magma", origin="lower", alpha=0.3)
            
            y_coords, x_coords = np.indices(sx.shape)
            ax1.quiver(x_coords, y_coords, sx, sy, color=ACCENT_COLOR, scale=0.5, width=0.005)
            ax1.set_title("DIFFERENTIAL SLOPES", color=TEXT_COLOR, fontsize=10, fontweight="bold")
        else:
            ax1.text(0.5, 0.5, "NO SLOPE DATA", ha="center", va="center", color=SECONDARY_TEXT)
        ax1.axis("off")

        # 3. Zernike Comparison
        ax2 = axes[1, 0]
        x_axis = np.arange(len(result.mode_labels))
        width = 0.35
        
        if result.hcipy_zernikes is not None:
            ax2.bar(x_axis - width/2, result.hcipy_zernikes * 1e9, width, label="HCIPy", color="#34495E", alpha=0.8)
        
        if result.fpga_zernikes is not None:
            ax2.bar(x_axis + width/2, result.fpga_zernikes * 1e9, width, label="FPGA", color=ACCENT_COLOR)
            
        ax2.set_xticks(x_axis)
        ax2.set_xticklabels(result.mode_labels, rotation=45, ha="right", fontsize=8)
        ax2.set_ylabel("Amplitude [nm]")
        ax2.set_title("ZERNIKE COEFFICIENTS", color=TEXT_COLOR, fontsize=10, fontweight="bold")
        ax2.legend(fontsize=8, framealpha=0.1)
        ax2.axhline(0, color=TEXT_COLOR, linewidth=0.5, alpha=0.3)

        # 4. Error distribution or PSF
        ax3 = axes[1, 1]
        if result.has_accuracy_comparison:
            errors = (result.fpga_zernikes - result.hcipy_zernikes) * 1e9
            ax3.stem(x_axis, errors, linefmt=f"{ERROR_COLOR}-", markerfmt=f"{ERROR_COLOR}o", basefmt="w-")
            ax3.set_title("RECONSTRUCTION ERROR [nm]", color=TEXT_COLOR, fontsize=10, fontweight="bold")
            ax3.set_xticks(x_axis)
            ax3.set_xticklabels(result.mode_labels, rotation=45, ha="right", fontsize=8)
        else:
            ax3.text(0.5, 0.5, "AWAITING COMPARISON", ha="center", va="center", color=SECONDARY_TEXT)
        
        self.draw_idle()

class AoDemoCanvas(FigureCanvasQTAgg):
    """Specialized canvas for the AO Imaging Demo."""
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        self.figure = Figure(figsize=(12, 10), facecolor=BG_COLOR)
        super().__init__(self.figure)
        self.setParent(parent)

    def update_result(self, result: RunResult) -> None:
        self.figure.clear()
        if result.fpga_zernikes is None or result.hcipy_zernikes is None:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "RUN SIMULATION TO GENERATE AO DEMO", ha="center", va="center", color=SECONDARY_TEXT, fontsize=14)
            ax.axis("off")
            self.draw_idle()
            return

        try:
            # Reconstruct phase using shwfs_utils helpers
            # We'll use a simplified 128x128 grid for speed in GUI
            side = 128
            pupil_grid = shwfs_utils.hcipy.make_pupil_grid(side, 1.0)
            aperture = shwfs_utils.hcipy.circular_aperture(0.9)(pupil_grid)
            
            z_basis = shwfs_utils.hcipy.make_zernike_basis(len(result.mode_labels), 1.0, pupil_grid, 1)
            
            # Reconstruction phase fields
            phase_hcipy = z_basis.linear_combination(result.hcipy_zernikes)
            phase_fpga = z_basis.linear_combination(result.fpga_zernikes)
            
            # Propagation to PSF
            wf_ref = shwfs_utils.hcipy.Wavefront(aperture, 1.0)
            wf_aber = shwfs_utils.hcipy.Wavefront(aperture * np.exp(1j * (phase_hcipy)), 1.0)
            wf_corr = shwfs_utils.hcipy.Wavefront(aperture * np.exp(1j * (phase_hcipy - phase_fpga)), 1.0)
            
            prop = shwfs_utils.hcipy.FraunhoferPropagator(pupil_grid, shwfs_utils.hcipy.make_focal_grid(8, 4))
            
            psf_ref = prop(wf_ref).intensity
            psf_aber = prop(wf_aber).intensity
            psf_corr = prop(wf_corr).intensity
            
            # Normalization
            peak = psf_ref.max()
            strehl_aber = psf_aber.max() / peak
            strehl_corr = psf_corr.max() / peak
            
            axes = self.figure.subplots(1, 2)
            self.figure.suptitle(f"STREHL RATIO IMPROVEMENT: {strehl_aber:.2f} \u2192 {strehl_corr:.2f}", color=SUCCESS_COLOR, fontsize=16, fontweight="bold")
            
            # Aberrated PSF
            ax0 = axes[0]
            shwfs_utils.hcipy.imshow_field(np.log10(psf_aber / peak + 1e-6), ax=ax0, cmap="magma", vmin=-4, vmax=0)
            ax0.set_title("UNCORRECTED PSF (LOG SCALE)", color=TEXT_COLOR, fontsize=10)
            ax0.axis("off")
            
            # Corrected PSF
            ax1 = axes[1]
            shwfs_utils.hcipy.imshow_field(np.log10(psf_corr / peak + 1e-6), ax=ax1, cmap="magma", vmin=-4, vmax=0)
            ax1.set_title("FPGA CORRECTED PSF (LOG SCALE)", color=TEXT_COLOR, fontsize=10)
            ax1.axis("off")
            
        except Exception as e:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f"ERROR GENERATING PSF: {e}", ha="center", va="center", color=ERROR_COLOR)
            ax.axis("off")

        self.draw_idle()

# ===========================================================================
# MAIN TABS
# ===========================================================================

class DashboardTab(QtWidgets.QWidget):
    start_server_requested = QtCore.Signal(str, int)
    stop_server_requested = QtCore.Signal()
    preview_requested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        
        # Left Panel: Controls & Status
        left_panel = QtWidgets.QWidget()
        left_panel.setFixedWidth(350)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        
        # 1. Connection Controls
        conn_box = QtWidgets.QGroupBox("SYSTEM CONTROL")
        conn_layout = QtWidgets.QGridLayout(conn_box)
        
        self.host_edit = QtWidgets.QLineEdit(DEFAULT_BIND_IP)
        self.port_spin = QtWidgets.QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(DEFAULT_BIND_PORT)
        
        self.start_btn = QtWidgets.QPushButton("ARM SERVER")
        self.stop_btn = QtWidgets.QPushButton("STOP")
        self.preview_btn = QtWidgets.QPushButton("LOCAL SIMULATION")
        
        self.start_btn.setStyleSheet(f"background-color: {ACCENT_COLOR}; color: {BG_COLOR}; font-weight: bold;")
        self.stop_btn.setStyleSheet(f"background-color: {ERROR_COLOR}; color: white;")
        self.preview_btn.setStyleSheet(f"background-color: {SURFACE_COLOR}; border: 1px solid {ACCENT_COLOR}; color: {ACCENT_COLOR};")

        conn_layout.addWidget(QtWidgets.QLabel("HOST"), 0, 0)
        conn_layout.addWidget(self.host_edit, 0, 1)
        conn_layout.addWidget(QtWidgets.QLabel("PORT"), 1, 0)
        conn_layout.addWidget(self.port_spin, 1, 1)
        conn_layout.addWidget(self.start_btn, 2, 0, 1, 2)
        conn_layout.addWidget(self.stop_btn, 3, 0, 1, 2)
        conn_layout.addWidget(self.preview_btn, 4, 0, 1, 2)
        
        # 2. Status Cards
        self.server_status = InfoCard("SERVER STATE", "Stopped")
        self.client_status = InfoCard("CONNECTION", "Disconnected")
        self.run_status = InfoCard("LAST RUN", "No Data")
        
        # 3. Log
        self.log_output = QtWidgets.QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet(f"background-color: {BG_COLOR}; border: 1px solid {BORDER_COLOR}; color: {SECONDARY_TEXT}; font-family: 'Consolas', 'Monaco', monospace; font-size: 10px;")
        
        left_layout.addWidget(conn_box)
        left_layout.addWidget(self.server_status)
        left_layout.addWidget(self.client_status)
        left_layout.addWidget(self.run_status)
        left_layout.addWidget(QtWidgets.QLabel("SYSTEM LOGS"))
        left_layout.addWidget(self.log_output, 1)
        
        # Right Panel: Visualization
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        self.canvas = PlotCanvas()
        right_layout.addWidget(self.canvas)
        
        # Main Layout
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)

        # Signals
        self.start_btn.clicked.connect(self._emit_start)
        self.stop_btn.clicked.connect(self.stop_server_requested.emit)
        self.preview_btn.clicked.connect(self.preview_requested.emit)

    def _emit_start(self) -> None:
        self.start_server_requested.emit(self.host_edit.text().strip(), int(self.port_spin.value()))

    def update_result(self, result: RunResult) -> None:
        self.run_status.set_value(f"{result.source.upper()}\n{result.timestamp}")
        self.canvas.update_result(result)
        
    def append_log(self, message: str) -> None:
        self.log_output.appendPlainText(f"[{QtCore.QTime.currentTime().toString()}] {message}")

class AccuracyTab(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # Metrics Row
        metrics_layout = QtWidgets.QHBoxLayout()
        self.rmse_card = InfoCard("RMSE (nm)")
        self.mae_card = InfoCard("MAE (nm)")
        self.max_card = InfoCard("MAX ERROR (nm)")
        self.corr_card = InfoCard("CORRELATION")
        
        metrics_layout.addWidget(self.rmse_card)
        metrics_layout.addWidget(self.mae_card)
        metrics_layout.addWidget(self.max_card)
        metrics_layout.addWidget(self.corr_card)
        
        # Table
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["MODE", "HCIPY (nm)", "FPGA (nm)", "ABS ERR (nm)", "REL ERR %"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {SURFACE_COLOR};
                gridline-color: {BORDER_COLOR};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
            }}
            QHeaderView::section {{
                background-color: {BORDER_COLOR};
                color: {ACCENT_COLOR};
                font-weight: bold;
                padding: 6px;
                border: none;
            }}
        """)
        
        layout.addLayout(metrics_layout)
        layout.addWidget(self.table, 1)

    def update_result(self, result: RunResult) -> None:
        if not result.has_accuracy_comparison:
            return

        baseline = np.asarray(result.hcipy_zernikes, dtype=np.float64) * 1e9
        measured = np.asarray(result.fpga_zernikes, dtype=np.float64) * 1e9
        errors = measured - baseline
        abs_errors = np.abs(errors)
        
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(abs_errors)
        max_err = np.max(abs_errors)
        corr = np.corrcoef(baseline, measured)[0, 1] if baseline.size > 1 else 1.0
        
        self.rmse_card.set_value(f"{rmse:.3f}")
        self.mae_card.set_value(f"{mae:.3f}")
        self.max_card.set_value(f"{max_err:.3f}")
        self.corr_card.set_value(f"{corr:.5f}")
        
        self.table.setRowCount(len(result.mode_labels))
        for i, label in enumerate(result.mode_labels):
            rel_err = (abs_errors[i] / abs(baseline[i]) * 100) if abs(baseline[i]) > 1e-6 else 0.0
            row_data = [
                label,
                f"{baseline[i]:.3f}",
                f"{measured[i]:.3f}",
                f"{abs_errors[i]:.3f}",
                f"{rel_err:.2f}%"
            ]
            for col, text in enumerate(row_data):
                item = QtWidgets.QTableWidgetItem(text)
                if col > 0:
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(i, col, item)

# ===========================================================================
# MAIN WINDOW
# ===========================================================================

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FPGA SHWFS CONTROL TERMINAL v3.0")
        self.resize(1400, 950)
        self.setAcceptDrops(True)
        
        # Service
        self.service = FpgaControlService(self)
        
        # Tabs
        self.tabs = QtWidgets.QTabWidget()
        self.dashboard = DashboardTab()
        self.accuracy = AccuracyTab()
        self.ao_demo = AoDemoCanvas()
        
        self.tabs.addTab(self.dashboard, "DASHBOARD")
        self.tabs.addTab(self.accuracy, "ACCURACY METRICS")
        self.tabs.addTab(self.ao_demo, "AO IMAGING DEMO")
        
        self.setCentralWidget(self.tabs)
        
        # Dark Theme Styling
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {BG_COLOR};
            }}
            QTabWidget::pane {{
                border: 1px solid {BORDER_COLOR};
                background: {BG_COLOR};
            }}
            QTabBar::tab {{
                background: {SURFACE_COLOR};
                color: {SECONDARY_TEXT};
                padding: 12px 30px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 2px;
                font-weight: bold;
                font-size: 11px;
            }}
            QTabBar::tab:selected {{
                background: {BG_COLOR};
                color: {ACCENT_COLOR};
                border-bottom: 2px solid {ACCENT_COLOR};
            }}
            QGroupBox {{
                color: {ACCENT_COLOR};
                font-weight: bold;
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
            QLabel {{
                color: {TEXT_COLOR};
            }}
            QLineEdit, QSpinBox {{
                background: {SURFACE_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 4px;
                color: {TEXT_COLOR};
                padding: 5px;
            }}
        """)
        
        # Signal Connections
        self.dashboard.start_server_requested.connect(self.service.start_server)
        self.dashboard.stop_server_requested.connect(self.service.stop_server)
        self.dashboard.preview_requested.connect(self.service.generate_local_preview)
        
        self.service.log_message.connect(self.dashboard.append_log)
        self.service.server_state_changed.connect(self.dashboard.server_status.set_value)
        self.service.client_state_changed.connect(self.dashboard.client_status.set_value)
        self.service.run_completed.connect(self._handle_result)
        self.service.error_reported.connect(self._handle_error)

        self.statusBar().showMessage("SYSTEM READY")
        self.statusBar().setStyleSheet(f"color: {SECONDARY_TEXT}; border-top: 1px solid {BORDER_COLOR};")

    def _handle_result(self, result: RunResult) -> None:
        self.dashboard.update_result(result)
        self.accuracy.update_result(result)
        self.ao_demo.update_result(result)
        self.statusBar().showMessage(f"RUN COMPLETED: {result.source.upper()} at {result.timestamp}", 5000)

    def _handle_error(self, message: str) -> None:
        QtWidgets.QMessageBox.critical(self, "SYSTEM ERROR", message)
        self.dashboard.append_log(f"ERROR: {message}")

def main() -> int:
    # High DPI scaling
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
    
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Custom Palette for dark mode fallback
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorGroup.All, QtGui.QPalette.ColorRole.Window, QtGui.QColor(BG_COLOR))
    palette.setColor(QtGui.QPalette.ColorGroup.All, QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(TEXT_COLOR))
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())