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
    NUM_SUBAPERTURES_SIDE,
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
        self.title_label.setStyleSheet(f"color: {SECONDARY_TEXT}; font-size: 11px; font-weight: bold; letter-spacing: 1px;")
        
        self.value_label = QtWidgets.QLabel(value)
        self.value_label.setStyleSheet(f"color: {ACCENT_COLOR}; font-size: 20px; font-weight: bold;")
        self.value_label.setWordWrap(True)
        
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addStretch()

    def set_value(self, value: str, color: str = ACCENT_COLOR) -> None:
        self.value_label.setText(value)
        self.value_label.setStyleSheet(f"color: {color}; font-size: 20px; font-weight: bold;")

class WfsAnalysisCanvas(FigureCanvasQTAgg):
    """Replicates the 2x3 HCIPy SHWFS visualization grid."""
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        self.figure = Figure(figsize=(16, 10), facecolor=BG_COLOR)
        super().__init__(self.figure)
        self.setParent(parent)
        plt.style.use("dark_background")

    def update_result(self, result: RunResult) -> None:
        self.figure.clear()
        if not result.simulation_data:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "NO SIMULATION DATA TO DISPLAY", ha="center", va="center", color=SECONDARY_TEXT, fontsize=16)
            ax.axis("off")
            self.draw_idle()
            return

        axes = self.figure.subplots(2, 3)
        self.figure.subplots_adjust(hspace=0.3, wspace=0.3)
        sim = result.simulation_data
        
        # Unpack needed fields
        image_ref = sim["image_ref"]
        image_aber = np.asarray(result.quantized_image)
        if image_aber.ndim == 1:
            side = int(np.sqrt(image_aber.size))
            image_aber = image_aber.reshape(side, side)
            
        aperture = sim["aperture"]
        input_opd_field = sim["input_opd_field"]
        
        # Use HCIPy's valid subaperture positions for plotting vector field
        estimator = sim["estimator"]
        shwfs_optics = sim["shwfs"]
        sub_positions = shwfs_optics.mla_grid.subset(estimator.estimation_subapertures)
        
        # FPGA slope vectors
        if result.slopes_xy_grid is not None:
            # Flattens back to only valid subapertures matching HCIPy format
            mask = sim["valid_subaperture_mask"].reshape(NUM_SUBAPERTURES_SIDE, NUM_SUBAPERTURES_SIDE)
            sx = result.slopes_xy_grid[..., 0][mask]
            sy = result.slopes_xy_grid[..., 1][mask]
        else:
            sx = np.zeros(len(sub_positions))
            sy = np.zeros(len(sub_positions))

        # Reconstructed OPD (FPGA)
        z_basis = shwfs_utils.make_zernike_basis(len(result.mode_labels), 1.0, sim["pupil_grid"], 1)
        z_basis.grid = sim["pupil_grid"] # Ensure correct grid
        # Use true basis from sim if available
        if "zernike_modes" in sim:
            reconstructed_zernike = sum(c * m for c, m in zip(result.fpga_zernikes, sim["zernike_modes"]))
            reconstructed_opd_field = aperture * reconstructed_zernike
        else:
            reconstructed_opd_field = input_opd_field * 0 # Fallback

        # Row 0: Images
        ax00 = axes[0, 0]
        shwfs_utils.imshow_field(image_ref, ax=ax00, cmap="inferno")
        ax00.set_title("SHWFS image — flat wavefront", color=TEXT_COLOR, fontsize=10)

        ax01 = axes[0, 1]
        ax01.imshow(image_aber, cmap="inferno", origin="lower")
        ax01.set_title("SHWFS image — aberrated wavefront", color=TEXT_COLOR, fontsize=10)

        ax02 = axes[0, 2]
        ax02.imshow(image_aber, cmap="inferno", origin="lower", alpha=0.4)
        if len(sx) > 0 and len(sx) == len(sub_positions.x):
            pitch = shwfs_utils._estimate_subaperture_pitch(sub_positions, NUM_SUBAPERTURES_SIDE)
            max_magnitude = np.hypot(sx, sy).max() + 1e-30
            arrow_scale = pitch * 1.5 / max_magnitude
            ax02.quiver(
                sub_positions.x, sub_positions.y, 
                sx * arrow_scale, sy * arrow_scale, 
                color=ACCENT_COLOR, scale=1, scale_units="xy", angles="xy", width=0.003
            )
        ax02.set_title("Differential slope field", color=TEXT_COLOR, fontsize=10)

        # Row 1: OPDs and Zernikes
        pupil_mask = aperture > 0.5
        vmax_nm = np.abs(input_opd_field[pupil_mask]).max() * 1e9

        ax10 = axes[1, 0]
        shwfs_utils.imshow_field(input_opd_field * 1e9, ax=ax10, cmap="RdBu", vmin=-vmax_nm, vmax=vmax_nm, mask=aperture)
        ax10.set_title("Input OPD [nm]", color=TEXT_COLOR, fontsize=10)

        ax11 = axes[1, 1]
        shwfs_utils.imshow_field(reconstructed_opd_field * 1e9, ax=ax11, cmap="RdBu", vmin=-vmax_nm, vmax=vmax_nm, mask=aperture)
        ax11.set_title("Reconstructed OPD [nm] (FPGA)", color=TEXT_COLOR, fontsize=10)

        ax12 = axes[1, 2]
        x_axis = np.arange(len(result.mode_labels))
        width = 0.25
        
        ax12.bar(x_axis - width, sim["true_coeffs"] * 1e9, width, label="True", color="steelblue")
        ax12.bar(x_axis, result.hcipy_zernikes * 1e9, width, label="HCIPy", color="lightgray")
        ax12.bar(x_axis + width, result.fpga_zernikes * 1e9, width, label="FPGA", color="tomato")
        
        ax12.set_xticks(x_axis)
        ax12.set_xticklabels(result.mode_labels, rotation=45, ha="right", fontsize=8, color=TEXT_COLOR)
        ax12.set_ylabel("Coefficient [nm]", color=TEXT_COLOR)
        ax12.set_title("Zernike coefficients", color=TEXT_COLOR, fontsize=10)
        ax12.legend(loc="upper right", fontsize=8, facecolor=SURFACE_COLOR)
        ax12.axhline(0, color=SECONDARY_TEXT, linewidth=0.8)
        
        # Color aesthetic adjustments for all axes
        for ax in axes.flat:
            ax.tick_params(colors=SECONDARY_TEXT)
            for spine in ax.spines.values():
                spine.set_color(BORDER_COLOR)

        self.draw_idle()

class AoDemoCanvas(FigureCanvasQTAgg):
    """Replicates the 2x3 HCIPy AO Imaging Demo grid."""
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        self.figure = Figure(figsize=(16, 10), facecolor=BG_COLOR)
        super().__init__(self.figure)
        self.setParent(parent)
        plt.style.use("dark_background")

    def update_result(self, result: RunResult) -> None:
        self.figure.clear()
        if not result.simulation_data:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "NO SIMULATION DATA TO DISPLAY", ha="center", va="center", color=SECONDARY_TEXT, fontsize=16)
            ax.axis("off")
            self.draw_idle()
            return

        sim = result.simulation_data
        try:
            # We use a downsampled grid to keep GUI snappy
            side = 128
            pupil_grid = shwfs_utils.make_pupil_grid(side, 1.0)
            aperture = shwfs_utils.make_obstructed_circular_aperture(1.0, 0.2)(pupil_grid)
            
            z_basis = shwfs_utils.make_zernike_basis(len(result.mode_labels), 1.0, pupil_grid, 1)
            
            # Reconstruction phase fields
            phase_aber = z_basis.linear_combination(sim["true_coeffs"])
            phase_fpga = z_basis.linear_combination(result.fpga_zernikes)
            
            # Propagation to PSF
            wf_ref = shwfs_utils.Wavefront(aperture, 1.0)
            wf_aber = shwfs_utils.Wavefront(aperture * np.exp(1j * phase_aber), 1.0)
            wf_corr = shwfs_utils.Wavefront(aperture * np.exp(1j * (phase_aber - phase_fpga)), 1.0)
            
            focal_grid = shwfs_utils.make_focal_grid(q=4, num_airy=10)
            prop = shwfs_utils.FraunhoferPropagator(pupil_grid, focal_grid)
            
            psf_ref = prop(wf_ref).intensity
            psf_aber = prop(wf_aber).intensity
            psf_corr = prop(wf_corr).intensity
            
            # Normalization
            peak = psf_ref.max()
            strehl_aber = psf_aber.max() / peak
            strehl_corr = psf_corr.max() / peak
            
            axes = self.figure.subplots(2, 3)
            self.figure.suptitle(f"AO IMAGING DEMO | STREHL RATIO: {strehl_aber:.3f} \u2192 {strehl_corr:.3f}", 
                                 color=SUCCESS_COLOR, fontsize=16, fontweight="bold")
            
            # Scene Convolution
            scene = shwfs_utils.load_vlt_demo_scene(side)
            psf_ref_2d = np.asarray(psf_ref).reshape(focal_grid.shape)
            psf_aber_2d = np.asarray(psf_aber).reshape(focal_grid.shape)
            psf_corr_2d = np.asarray(psf_corr).reshape(focal_grid.shape)
            
            # resize psf to scene size to convolve
            import scipy.ndimage as ndimage
            zoom_f = side / psf_ref_2d.shape[0]
            psf_r_res = ndimage.zoom(psf_ref_2d, zoom_f)
            psf_a_res = ndimage.zoom(psf_aber_2d, zoom_f)
            psf_c_res = ndimage.zoom(psf_corr_2d, zoom_f)
            
            img_dl = shwfs_utils._convolve_with_psf(scene, psf_r_res)
            img_ab = shwfs_utils._convolve_with_psf(scene, psf_a_res)
            img_corr = shwfs_utils._convolve_with_psf(scene, psf_c_res)
            
            img_dl /= img_dl.max() + 1e-12
            img_ab /= img_ab.max() + 1e-12
            img_corr /= img_corr.max() + 1e-12

            # Top row: Scenes
            axes[0, 0].imshow(scene, cmap="gray", origin="lower")
            axes[0, 0].set_title("Reference object (VLT image)", color=TEXT_COLOR, fontsize=10)
            axes[0, 0].axis("off")
            
            axes[0, 1].imshow(img_ab, cmap="gray", origin="lower")
            axes[0, 1].set_title("Aberrated image", color=TEXT_COLOR, fontsize=10)
            axes[0, 1].axis("off")
            
            axes[0, 2].imshow(img_corr, cmap="gray", origin="lower")
            axes[0, 2].set_title("Corrected image (FPGA OPD)", color=TEXT_COLOR, fontsize=10)
            axes[0, 2].axis("off")
            
            # Bottom row: PSFs
            axes[1, 0].imshow(np.log10(psf_r_res / psf_r_res.max() + 1e-5), cmap="magma", origin="lower")
            axes[1, 0].set_title("log10 PSF (diffraction-limited)", color=TEXT_COLOR, fontsize=10)
            axes[1, 0].axis("off")
            
            axes[1, 1].imshow(np.log10(psf_a_res / psf_r_res.max() + 1e-5), cmap="magma", origin="lower")
            axes[1, 1].set_title("log10 PSF (aberrated)", color=TEXT_COLOR, fontsize=10)
            axes[1, 1].axis("off")
            
            axes[1, 2].imshow(np.log10(psf_c_res / psf_r_res.max() + 1e-5), cmap="magma", origin="lower")
            axes[1, 2].set_title("log10 PSF (corrected)", color=TEXT_COLOR, fontsize=10)
            axes[1, 2].axis("off")
            
        except Exception as e:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f"ERROR GENERATING PSF:\n{e}", ha="center", va="center", color=ERROR_COLOR)
            ax.axis("off")

        self.draw_idle()


class PipelineCanvas(FigureCanvasQTAgg):
    """Static diagram of the FPGA processing pipeline."""
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        self.figure = Figure(figsize=(12, 8), facecolor=BG_COLOR)
        super().__init__(self.figure)
        self.setParent(parent)
        self.draw_pipeline()

    def draw_pipeline(self) -> None:
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(BG_COLOR)
        ax.axis("off")
        
        stages = [
            "Ethernet/HPS\nDetector Image\n(8-bit stream)",
            "TCoG Accumulator\ns(I), s(xI), s(yI)\n(20-bit/16-bit)",
            "Reciprocal\nNewton Raphson\n(27-bit Q0.27)",
            "Slope Calc\nCentroid - Ref\n(27-bit signed)",
            "E-Matrix MAC\n10 modes x 336\n(55-bit internal)",
            "Zernike Out\nModes 1-10\n(27-bit Q4.23)"
        ]
        
        x_centers = np.linspace(0.1, 0.9, len(stages))
        y_center = 0.5
        box_w = 0.12
        box_h = 0.2
        
        for i, (x, text) in enumerate(zip(x_centers, stages)):
            # Draw box
            rect = plt.Rectangle((x - box_w/2, y_center - box_h/2), box_w, box_h, 
                                 facecolor=SURFACE_COLOR, edgecolor=ACCENT_COLOR, 
                                 linewidth=2, zorder=2)
            ax.add_patch(rect)
            
            # Text inside
            ax.text(x, y_center, text, ha="center", va="center", color=TEXT_COLOR, 
                    fontsize=9, fontweight="bold", zorder=3)
            
            # Arrows between
            if i < len(stages) - 1:
                ax.annotate("", xy=(x_centers[i+1] - box_w/2, y_center), 
                            xytext=(x + box_w/2, y_center),
                            arrowprops=dict(arrowstyle="->", color=ACCENT_COLOR, lw=2), zorder=1)
        
        ax.set_title("FPGA DATAPATH PIPELINE", color=ACCENT_COLOR, fontsize=16, fontweight="bold", pad=20)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
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
        
        self.start_btn.setStyleSheet(f"background-color: {ACCENT_COLOR}; color: {BG_COLOR}; font-weight: bold; padding: 8px; border-radius: 4px;")
        self.stop_btn.setStyleSheet(f"background-color: {ERROR_COLOR}; color: white; padding: 8px; border-radius: 4px;")
        self.preview_btn.setStyleSheet(f"background-color: transparent; border: 1px solid {ACCENT_COLOR}; color: {ACCENT_COLOR}; padding: 8px; border-radius: 4px; font-weight: bold;")

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
        self.log_output.setStyleSheet(f"background-color: {BG_COLOR}; border: 1px solid {BORDER_COLOR}; color: {SECONDARY_TEXT}; font-family: 'Consolas', 'Monaco', monospace; font-size: 11px; padding: 5px;")
        
        left_layout.addWidget(conn_box)
        left_layout.addWidget(self.server_status)
        left_layout.addWidget(self.client_status)
        left_layout.addWidget(self.run_status)
        left_layout.addWidget(QtWidgets.QLabel("SYSTEM LOGS", styleSheet=f"color: {ACCENT_COLOR}; font-weight: bold;"))
        left_layout.addWidget(self.log_output, 1)
        
        # Right Panel: Primary Visualization (Subset)
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        # We can just show a simpler dashboard plot here or reuse WFS analysis
        self.canvas = FigureCanvasQTAgg(Figure(figsize=(8, 8), facecolor=BG_COLOR))
        self.ax_img = self.canvas.figure.add_subplot(211)
        self.ax_bar = self.canvas.figure.add_subplot(212)
        self.canvas.figure.subplots_adjust(hspace=0.4)
        self.ax_img.axis("off")
        self.ax_img.text(0.5, 0.5, "SYSTEM IDLE", ha="center", va="center", color=SECONDARY_TEXT, fontsize=20)
        self.ax_bar.axis("off")
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
        color = SUCCESS_COLOR if result.source == "fpga" else ACCENT_COLOR
        self.run_status.set_value(f"{result.source.upper()}\n{result.timestamp}", color)
        
        # Update quick dashboard view
        self.ax_img.clear()
        self.ax_bar.clear()
        
        img = np.asarray(result.quantized_image)
        if img.ndim == 1:
            side = int(np.sqrt(img.size))
            img = img.reshape(side, side)
            
        self.ax_img.imshow(img, cmap="inferno", origin="lower")
        self.ax_img.set_title("LATEST DETECTOR FRAME", color=TEXT_COLOR)
        self.ax_img.axis("off")
        
        x = np.arange(len(result.mode_labels))
        w = 0.35
        self.ax_bar.bar(x - w/2, result.hcipy_zernikes * 1e9, w, label="HCIPy", color="lightgray")
        self.ax_bar.bar(x + w/2, result.fpga_zernikes * 1e9, w, label="FPGA", color="tomato")
        self.ax_bar.set_xticks(x)
        self.ax_bar.set_xticklabels(result.mode_labels, rotation=45, ha="right", color=TEXT_COLOR)
        self.ax_bar.set_title("ZERNIKE COEFFICIENTS [nm]", color=TEXT_COLOR)
        self.ax_bar.legend()
        for spine in self.ax_bar.spines.values():
            spine.set_color(BORDER_COLOR)
        self.ax_bar.tick_params(colors=SECONDARY_TEXT)
        
        self.canvas.draw_idle()
        
    def append_log(self, message: str) -> None:
        if "ERROR" in message:
            fmt = f"<span style='color:{ERROR_COLOR}'>"
        elif "SUCCESS" in message or "finished" in message:
            fmt = f"<span style='color:{SUCCESS_COLOR}'>"
        else:
            fmt = f"<span>"
            
        self.log_output.appendHtml(f"<span style='color:{SECONDARY_TEXT}'>[{QtCore.QTime.currentTime().toString()}]</span> {fmt}{message}</span>")

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
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["MODE", "TRUE (nm)", "HCIPY (nm)", "FPGA (nm)", "FPGA ABS ERR", "FPGA REL ERR %"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {SURFACE_COLOR};
                gridline-color: {BORDER_COLOR};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                font-size: 14px;
            }}
            QHeaderView::section {{
                background-color: {BORDER_COLOR};
                color: {ACCENT_COLOR};
                font-weight: bold;
                padding: 10px;
                border: none;
            }}
            QTableWidget::item {{
                padding: 5px;
            }}
        """)
        
        layout.addLayout(metrics_layout)
        layout.addWidget(self.table, 1)

    def update_result(self, result: RunResult) -> None:
        if not result.has_accuracy_comparison:
            return

        baseline = np.asarray(result.true_coeffs, dtype=np.float64) * 1e9
        hcipy_vals = np.asarray(result.hcipy_zernikes, dtype=np.float64) * 1e9
        fpga_vals = np.asarray(result.fpga_zernikes, dtype=np.float64) * 1e9
        
        errors = fpga_vals - baseline
        abs_errors = np.abs(errors)
        
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(abs_errors)
        max_err = np.max(abs_errors)
        corr = np.corrcoef(baseline, fpga_vals)[0, 1] if baseline.size > 1 else 1.0
        
        color = SUCCESS_COLOR if rmse < 10.0 else ERROR_COLOR
        self.rmse_card.set_value(f"{rmse:.3f}", color)
        self.mae_card.set_value(f"{mae:.3f}")
        self.max_card.set_value(f"{max_err:.3f}")
        self.corr_card.set_value(f"{corr:.5f}")
        
        self.table.setRowCount(len(result.mode_labels))
        for i, label in enumerate(result.mode_labels):
            rel_err = (abs_errors[i] / abs(baseline[i]) * 100) if abs(baseline[i]) > 1e-6 else 0.0
            row_data = [
                label,
                f"{baseline[i]:.3f}",
                f"{hcipy_vals[i]:.3f}",
                f"{fpga_vals[i]:.3f}",
                f"{abs_errors[i]:.3f}",
                f"{rel_err:.2f}%"
            ]
            for col, text in enumerate(row_data):
                item = QtWidgets.QTableWidgetItem(text)
                if col > 0:
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                    if col >= 4:
                        item.setForeground(QtGui.QColor(ERROR_COLOR if abs_errors[i] > 10.0 else SUCCESS_COLOR))
                self.table.setItem(i, col, item)

# ===========================================================================
# MAIN WINDOW
# ===========================================================================

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FPGA SHWFS CONTROL TERMINAL v3.0 | PREMIUM EDITION")
        self.resize(1600, 1000)
        
        # Service
        self.service = FpgaControlService(self)
        
        # Tabs
        self.tabs = QtWidgets.QTabWidget()
        self.dashboard = DashboardTab()
        self.wfs_analysis = WfsAnalysisCanvas()
        self.ao_demo = AoDemoCanvas()
        self.accuracy = AccuracyTab()
        self.pipeline = PipelineCanvas()
        
        self.tabs.addTab(self.dashboard, "① DASHBOARD")
        self.tabs.addTab(self.wfs_analysis, "② WFS ANALYSIS")
        self.tabs.addTab(self.ao_demo, "③ AO IMAGING DEMO")
        self.tabs.addTab(self.accuracy, "④ ACCURACY METRICS")
        self.tabs.addTab(self.pipeline, "⑤ PIPELINE DIAGRAM")
        
        # Header
        header = QtWidgets.QLabel("SHACK-HARTMANN FPGA TELEMETRY")
        header.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet(f"""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {SURFACE_COLOR}, stop:0.5 {BORDER_COLOR}, stop:1 {SURFACE_COLOR});
            color: {ACCENT_COLOR};
            font-size: 24px;
            font-weight: bold;
            padding: 15px;
            letter-spacing: 4px;
        """)
        
        central_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(header)
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(central_widget)
        
        # Dark Theme Styling
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {BG_COLOR};
            }}
            QTabWidget::pane {{
                border: none;
                border-top: 1px solid {BORDER_COLOR};
                background: {BG_COLOR};
            }}
            QTabBar::tab {{
                background: {SURFACE_COLOR};
                color: {SECONDARY_TEXT};
                padding: 15px 30px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 2px;
                font-weight: bold;
                font-size: 13px;
                letter-spacing: 1px;
            }}
            QTabBar::tab:selected {{
                background: {BG_COLOR};
                color: {SUCCESS_COLOR};
                border-top: 2px solid {SUCCESS_COLOR};
            }}
            QTabBar::tab:hover:!selected {{
                background: {BORDER_COLOR};
                color: {TEXT_COLOR};
            }}
            QGroupBox {{
                color: {ACCENT_COLOR};
                font-weight: bold;
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 15px;
                font-size: 12px;
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
                background: {BG_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 4px;
                color: {TEXT_COLOR};
                padding: 6px;
                font-weight: bold;
            }}
            QScrollBar:vertical {{
                border: none;
                background: {SURFACE_COLOR};
                width: 12px;
            }}
            QScrollBar::handle:vertical {{
                background: {BORDER_COLOR};
                border-radius: 6px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {SECONDARY_TEXT};
            }}
        """)
        
        # Signal Connections
        self.dashboard.start_server_requested.connect(self.service.start_server)
        self.dashboard.stop_server_requested.connect(self.service.stop_server)
        self.dashboard.preview_requested.connect(self.service.generate_local_preview)
        
        self.service.log_message.connect(self.dashboard.append_log)
        self.service.server_state_changed.connect(lambda s: self.dashboard.server_status.set_value(s, SUCCESS_COLOR if "Listening" in s else ACCENT_COLOR))
        self.service.client_state_changed.connect(lambda s: self.dashboard.client_status.set_value(s, SUCCESS_COLOR if "Connected" in s else SECONDARY_TEXT))
        self.service.run_completed.connect(self._handle_result)
        self.service.error_reported.connect(self._handle_error)

        self.statusBar().showMessage("SYSTEM READY")
        self.statusBar().setStyleSheet(f"color: {SECONDARY_TEXT}; background: {SURFACE_COLOR}; border-top: 1px solid {BORDER_COLOR}; padding: 5px;")

    def _handle_result(self, result: RunResult) -> None:
        self.dashboard.update_result(result)
        self.wfs_analysis.update_result(result)
        self.ao_demo.update_result(result)
        self.accuracy.update_result(result)
        self.statusBar().showMessage(f"RUN COMPLETED: {result.source.upper()} at {result.timestamp}", 5000)

    def _handle_error(self, message: str) -> None:
        QtWidgets.QMessageBox.critical(self, "SYSTEM ERROR", message)
        self.dashboard.append_log(f"ERROR: {message}")

def main() -> int:
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
    
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Custom Palette for dark mode fallback
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorGroup.All, QtGui.QPalette.ColorRole.Window, QtGui.QColor(BG_COLOR))
    palette.setColor(QtGui.QPalette.ColorGroup.All, QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(TEXT_COLOR))
    app.setPalette(palette)
    
    # Force use of modern fonts if available
    font = QtGui.QFont("Inter", 10)
    font.setStyleHint(QtGui.QFont.StyleHint.SansSerif)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())