from __future__ import annotations

import os
import sys
from collections.abc import MutableMapping
from typing import cast

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from fpga_gui_service import (
    DEFAULT_BIND_IP,
    DEFAULT_BIND_PORT,
    FPGA_RM_GLOBAL_SCALE,
    FpgaControlService,
    RunResult,
)


class ResultsTab(QtWidgets.QWidget):
    start_server_requested = QtCore.Signal(str, int)
    stop_server_requested = QtCore.Signal()
    preview_requested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self.server_state_value = QtWidgets.QLabel("Stopped")
        self.client_state_value = QtWidgets.QLabel("No client connected")
        self.last_run_value = QtWidgets.QLabel("No runs yet")
        self.note_value = QtWidgets.QLabel("Start listening for a hardware run or generate a local preview.")
        self.note_value.setWordWrap(True)

        self.host_edit = QtWidgets.QLineEdit(DEFAULT_BIND_IP)
        self.port_spin = QtWidgets.QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(DEFAULT_BIND_PORT)

        self.start_button = QtWidgets.QPushButton("Start Listening")
        self.stop_button = QtWidgets.QPushButton("Stop Server")
        self.preview_button = QtWidgets.QPushButton("Run Local Preview")

        self.start_button.clicked.connect(self._emit_start_server)
        self.stop_button.clicked.connect(self.stop_server_requested.emit)
        self.preview_button.clicked.connect(self.preview_requested.emit)

        self.coeff_table = QtWidgets.QTableWidget(0, 4)
        self.coeff_table.setHorizontalHeaderLabels(["Mode", "HCIPy (m)", "FPGA (m)", "Error (m)"])
        self.coeff_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.coeff_table.verticalHeader().setVisible(False)
        self.coeff_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.coeff_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)

        self.log_output = QtWidgets.QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumBlockCount(1000)

        controls_layout = QtWidgets.QHBoxLayout()
        controls_layout.addWidget(QtWidgets.QLabel("Bind IP"))
        controls_layout.addWidget(self.host_edit, 1)
        controls_layout.addWidget(QtWidgets.QLabel("Port"))
        controls_layout.addWidget(self.port_spin)
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.preview_button)

        status_layout = QtWidgets.QGridLayout()
        status_layout.addWidget(self._make_card("Server State", self.server_state_value), 0, 0)
        status_layout.addWidget(self._make_card("Client State", self.client_state_value), 0, 1)
        status_layout.addWidget(self._make_card("Last Run", self.last_run_value), 0, 2)

        guidance_box = QtWidgets.QGroupBox("Protocol Notes")
        guidance_layout = QtWidgets.QVBoxLayout(guidance_box)
        guidance_layout.addWidget(
            QtWidgets.QLabel(
                "The current TCP protocol is client-initiated. The HPS client sends start, "
                "so the host GUI primarily arms the server, shows connection state, and displays the latest run."
            )
        )
        guidance_layout.addWidget(self.note_value)

        coeff_box = QtWidgets.QGroupBox("Latest Coefficient Comparison")
        coeff_layout = QtWidgets.QVBoxLayout(coeff_box)
        coeff_layout.addWidget(self.coeff_table)

        log_box = QtWidgets.QGroupBox("Session Log")
        log_layout = QtWidgets.QVBoxLayout(log_box)
        log_layout.addWidget(self.log_output)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(controls_layout)
        layout.addLayout(status_layout)
        layout.addWidget(guidance_box)
        layout.addWidget(coeff_box, 1)
        layout.addWidget(log_box, 1)

    def _make_card(self, title: str, value_label: QtWidgets.QLabel) -> QtWidgets.QGroupBox:
        value_label.setObjectName("ValueCard")
        card = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout(card)
        layout.addWidget(value_label)
        return card

    def _emit_start_server(self) -> None:
        self.start_server_requested.emit(self.host_edit.text().strip(), int(self.port_spin.value()))

    def append_log(self, message: str) -> None:
        self.log_output.appendPlainText(message)

    def set_server_state(self, state: str) -> None:
        self.server_state_value.setText(state)

    def set_client_state(self, state: str) -> None:
        self.client_state_value.setText(state)

    def update_result(self, result: RunResult) -> None:
        suffix = f" via {result.client_address}" if result.client_address else ""
        self.last_run_value.setText(f"{result.source} at {result.timestamp}{suffix}")
        self.note_value.setText(" ".join(result.notes) if result.notes else "No additional notes.")

        baseline = result.hcipy_zernikes
        measured = result.fpga_zernikes
        row_count = len(result.mode_labels)
        self.coeff_table.setRowCount(row_count)
        for row_index, mode_label in enumerate(result.mode_labels):
            hcipy_text = "n/a"
            fpga_text = "n/a"
            error_text = "n/a"
            if baseline is not None and row_index < len(baseline):
                hcipy_text = f"{float(baseline[row_index]):+.6e}"
            if measured is not None and row_index < len(measured):
                fpga_text = f"{float(measured[row_index]):+.6e}"
            if baseline is not None and measured is not None and row_index < len(baseline) and row_index < len(measured):
                error_text = f"{float(measured[row_index] - baseline[row_index]):+.6e}"

            for column, text in enumerate((mode_label, hcipy_text, fpga_text, error_text)):
                item = QtWidgets.QTableWidgetItem(text)
                if column > 0:
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                self.coeff_table.setItem(row_index, column, item)


class AccuracyTab(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self.message_label = QtWidgets.QLabel("No FPGA run has completed yet.")
        self.message_label.setWordWrap(True)

        self.rmse_value = QtWidgets.QLabel("n/a")
        self.mae_value = QtWidgets.QLabel("n/a")
        self.max_error_value = QtWidgets.QLabel("n/a")
        self.correlation_value = QtWidgets.QLabel("n/a")

        self.metrics_table = QtWidgets.QTableWidget(0, 5)
        self.metrics_table.setHorizontalHeaderLabels(["Mode", "HCIPy (m)", "FPGA (m)", "Abs Error (m)", "Rel Error %"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.metrics_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)

        summary_layout = QtWidgets.QGridLayout()
        summary_layout.addWidget(self._make_metric_card("RMSE", self.rmse_value), 0, 0)
        summary_layout.addWidget(self._make_metric_card("Mean Abs Error", self.mae_value), 0, 1)
        summary_layout.addWidget(self._make_metric_card("Max Abs Error", self.max_error_value), 0, 2)
        summary_layout.addWidget(self._make_metric_card("Correlation", self.correlation_value), 0, 3)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.message_label)
        layout.addLayout(summary_layout)
        layout.addWidget(self.metrics_table, 1)

    def _make_metric_card(self, title: str, value_label: QtWidgets.QLabel) -> QtWidgets.QGroupBox:
        value_label.setObjectName("ValueCard")
        card = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout(card)
        layout.addWidget(value_label)
        return card

    def update_result(self, result: RunResult) -> None:
        if not result.has_accuracy_comparison:
            self.message_label.setText(
                "Local preview is available, but FPGA-vs-HCIPy accuracy metrics populate after a hardware run completes."
            )
            self.metrics_table.setRowCount(0)
            for label in (self.rmse_value, self.mae_value, self.max_error_value, self.correlation_value):
                label.setText("n/a")
            return

        baseline = np.asarray(result.hcipy_zernikes, dtype=np.float64)
        measured = np.asarray(result.fpga_zernikes, dtype=np.float64)
        errors = measured - baseline
        abs_errors = np.abs(errors)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_errors = np.where(np.abs(baseline) > 1e-12, abs_errors / np.abs(baseline) * 100.0, np.nan)

        rmse = float(np.sqrt(np.mean(errors**2)))
        mae = float(np.mean(abs_errors))
        max_error = float(np.max(abs_errors))
        correlation = float(np.corrcoef(baseline, measured)[0, 1]) if baseline.size > 1 else float("nan")

        self.message_label.setText(
            f"Accuracy metrics for the most recent FPGA run at {result.timestamp}."
        )
        self.rmse_value.setText(f"{rmse:.6e}")
        self.mae_value.setText(f"{mae:.6e}")
        self.max_error_value.setText(f"{max_error:.6e}")
        self.correlation_value.setText("n/a" if np.isnan(correlation) else f"{correlation:.5f}")

        self.metrics_table.setRowCount(len(result.mode_labels))
        for row_index, mode_label in enumerate(result.mode_labels):
            row_values = (
                mode_label,
                f"{baseline[row_index]:+.6e}",
                f"{measured[row_index]:+.6e}",
                f"{abs_errors[row_index]:.6e}",
                "n/a" if np.isnan(rel_errors[row_index]) else f"{rel_errors[row_index]:.3f}",
            )
            for column, text in enumerate(row_values):
                item = QtWidgets.QTableWidgetItem(text)
                if column > 0:
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                self.metrics_table.setItem(row_index, column, item)


class ComparisonCanvas(FigureCanvasQTAgg):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        self.figure = Figure(figsize=(10, 4), tight_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)

    def show_placeholder(self, message: str) -> None:
        self.figure.clear()
        axis = self.figure.subplots()
        axis.text(0.5, 0.5, message, ha="center", va="center")
        axis.set_axis_off()
        self.draw_idle()

    def update_result(self, result: RunResult) -> None:
        if not result.has_accuracy_comparison:
            self.show_placeholder("Comparison plots appear after an FPGA run completes.")
            return

        baseline_nm = np.asarray(result.hcipy_zernikes, dtype=np.float64) * 1e9
        measured_nm = np.asarray(result.fpga_zernikes, dtype=np.float64) * 1e9
        errors_nm = measured_nm - baseline_nm

        self.figure.clear()
        axes = self.figure.subplots(1, 2)

        combined = np.concatenate((baseline_nm, measured_nm))
        max_extent = max(1.0, float(np.max(np.abs(combined))) * 1.1)

        axes[0].scatter(baseline_nm, measured_nm, color="#e76f51", s=48)
        axes[0].plot([-max_extent, max_extent], [-max_extent, max_extent], linestyle="--", color="#264653")
        for index, label in enumerate(result.mode_labels):
            axes[0].annotate(label, (baseline_nm[index], measured_nm[index]), textcoords="offset points", xytext=(4, 4))
        axes[0].set_xlim(-max_extent, max_extent)
        axes[0].set_ylim(-max_extent, max_extent)
        axes[0].set_xlabel("HCIPy (nm)")
        axes[0].set_ylabel("FPGA (nm)")
        axes[0].set_title("Coefficient Correlation")
        axes[0].grid(alpha=0.25)

        x_axis = np.arange(len(result.mode_labels))
        colors = ["#e76f51" if value >= 0.0 else "#2a9d8f" for value in errors_nm]
        axes[1].bar(x_axis, errors_nm, color=colors)
        axes[1].axhline(0.0, color="#264653", linewidth=0.9)
        axes[1].set_xticks(x_axis)
        axes[1].set_xticklabels(result.mode_labels, rotation=45, ha="right")
        axes[1].set_ylabel("FPGA - HCIPy (nm)")
        axes[1].set_title("Signed Error By Mode")

        self.draw_idle()


class ComparisonTab(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self.message_label = QtWidgets.QLabel("No FPGA run has completed yet.")
        self.message_label.setWordWrap(True)

        self.scale_value = QtWidgets.QLabel(f"G = {FPGA_RM_GLOBAL_SCALE:g}")
        self.mean_bias_value = QtWidgets.QLabel("n/a")
        self.worst_mode_value = QtWidgets.QLabel("n/a")

        self.canvas = ComparisonCanvas(self)
        self.canvas.show_placeholder("Comparison plots appear after an FPGA run completes.")

        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Mode", "HCIPy (nm)", "FPGA (nm)", "Error (nm)", "FPGA Raw"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)

        summary_layout = QtWidgets.QGridLayout()
        summary_layout.addWidget(self._make_metric_card("RM Scale", self.scale_value), 0, 0)
        summary_layout.addWidget(self._make_metric_card("Mean Bias", self.mean_bias_value), 0, 1)
        summary_layout.addWidget(self._make_metric_card("Worst Mode", self.worst_mode_value), 0, 2)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.message_label)
        layout.addLayout(summary_layout)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.table, 1)

    def _make_metric_card(self, title: str, value_label: QtWidgets.QLabel) -> QtWidgets.QGroupBox:
        value_label.setObjectName("ValueCard")
        card = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout(card)
        layout.addWidget(value_label)
        return card

    def update_result(self, result: RunResult) -> None:
        self.scale_value.setText(f"G = {FPGA_RM_GLOBAL_SCALE:g}")
        if not result.has_accuracy_comparison:
            self.message_label.setText("HCIPy-vs-FPGA comparison populates after a hardware run completes.")
            self.mean_bias_value.setText("n/a")
            self.worst_mode_value.setText("n/a")
            self.table.setRowCount(0)
            self.canvas.show_placeholder("Comparison plots appear after an FPGA run completes.")
            return

        baseline_nm = np.asarray(result.hcipy_zernikes, dtype=np.float64) * 1e9
        measured_nm = np.asarray(result.fpga_zernikes, dtype=np.float64) * 1e9
        errors_nm = measured_nm - baseline_nm
        raw_values = result.fpga_zernikes_raw

        worst_index = int(np.argmax(np.abs(errors_nm)))
        self.message_label.setText(
            f"Comparing the HCIPy baseline against FPGA outputs from the run at {result.timestamp}."
        )
        self.mean_bias_value.setText(f"{float(np.mean(errors_nm)):+.3f} nm")
        self.worst_mode_value.setText(f"{result.mode_labels[worst_index]} ({errors_nm[worst_index]:+.3f} nm)")

        self.table.setRowCount(len(result.mode_labels))
        for row_index, mode_label in enumerate(result.mode_labels):
            raw_text = "n/a"
            if raw_values is not None and row_index < len(raw_values):
                raw_text = str(int(raw_values[row_index]))

            row_values = (
                mode_label,
                f"{baseline_nm[row_index]:+.3f}",
                f"{measured_nm[row_index]:+.3f}",
                f"{errors_nm[row_index]:+.3f}",
                raw_text,
            )
            for column, text in enumerate(row_values):
                item = QtWidgets.QTableWidgetItem(text)
                if column > 0:
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(row_index, column, item)

        self.canvas.update_result(result)


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        self.figure = Figure(figsize=(10, 8), tight_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)

    def _masked_component(self, xy_grid: np.ndarray, mask: np.ndarray | None, component_index: int) -> np.ma.MaskedArray:
        component = np.asarray(xy_grid[..., component_index], dtype=np.float64)
        if mask is None:
            return np.ma.array(component)
        return np.ma.array(component, mask=~np.asarray(mask, dtype=bool))

    def _plot_signed_field(
        self,
        axis: Axes,
        values: np.ma.MaskedArray,
        title: str,
        cmap: str = "RdBu_r",
    ) -> None:
        valid_values = np.asarray(values.compressed(), dtype=np.float64)
        vmax = float(np.max(np.abs(valid_values))) if valid_values.size else 1.0
        image = axis.imshow(values, cmap=cmap, origin="lower", vmin=-vmax, vmax=vmax)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        self.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    def _plot_coefficients(self, axis: Axes, result: RunResult) -> None:
        x_axis = np.arange(len(result.mode_labels))
        axis.axhline(0.0, color="#222222", linewidth=0.8)

        true_nm = None if result.true_zernikes is None else np.asarray(result.true_zernikes, dtype=np.float64) * 1e9
        hcipy_nm = None if result.hcipy_zernikes is None else np.asarray(result.hcipy_zernikes, dtype=np.float64) * 1e9
        fpga_nm = None if result.fpga_zernikes is None else np.asarray(result.fpga_zernikes, dtype=np.float64) * 1e9

        if true_nm is not None and hcipy_nm is not None and fpga_nm is not None:
            width = 0.24
            axis.bar(x_axis - width, true_nm, width=width, label="True", color="#457b9d")
            axis.bar(x_axis, hcipy_nm, width=width, label="HCIPy", color="#2a9d8f")
            axis.bar(x_axis + width, fpga_nm, width=width, label="FPGA", color="#e76f51")
            axis.set_title("Coefficient Comparison")
        elif true_nm is not None and hcipy_nm is not None:
            width = 0.36
            axis.bar(x_axis - width / 2, true_nm, width=width, label="True", color="#457b9d")
            axis.bar(x_axis + width / 2, hcipy_nm, width=width, label="HCIPy", color="#2a9d8f")
            axis.set_title("HCIPy vs True Coefficients")
        elif hcipy_nm is not None:
            axis.bar(x_axis, hcipy_nm, width=0.6, label="HCIPy", color="#2a9d8f")
            axis.set_title("HCIPy Coefficients")
        else:
            axis.text(0.5, 0.5, "No coefficient data", ha="center", va="center")
            axis.set_title("Coefficient View")
            axis.set_xticks([])
            axis.set_yticks([])
            return

        axis.set_xticks(x_axis)
        axis.set_xticklabels(result.mode_labels, rotation=45, ha="right")
        axis.set_ylabel("Coefficient [nm]")
        axis.legend(loc="upper right")

    def update_result(self, result: RunResult) -> None:
        self.figure.clear()
        axes = self.figure.subplots(2, 2)

        image = np.asarray(result.quantized_image)
        if image.ndim == 1:
            side = int(np.sqrt(image.size))
            image = image.reshape(side, side)

        axes[0, 0].imshow(image, cmap="inferno", origin="lower")
        axes[0, 0].set_title("Quantized detector frame")
        axes[0, 0].set_xticks([])
        axes[0, 0].set_yticks([])

        if result.slopes_xy_grid is not None:
            slope_x = self._masked_component(result.slopes_xy_grid, result.valid_subaperture_mask, 0)
            self._plot_signed_field(axes[0, 1], slope_x, "Slope X [px]")
        else:
            axes[0, 1].text(0.5, 0.5, "No slope data", ha="center", va="center")
            axes[0, 1].set_title("Slope X [px]")
            axes[0, 1].set_xticks([])
            axes[0, 1].set_yticks([])

        if result.slopes_xy_grid is not None:
            slope_y = self._masked_component(result.slopes_xy_grid, result.valid_subaperture_mask, 1)
            self._plot_signed_field(axes[1, 0], slope_y, "Slope Y [px]")
        else:
            axes[1, 0].text(0.5, 0.5, "No slope data", ha="center", va="center")
            axes[1, 0].set_title("Slope Y [px]")
            axes[1, 0].set_xticks([])
            axes[1, 0].set_yticks([])

        self._plot_coefficients(axes[1, 1], result)

        self.draw_idle()


class PlotsTab(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.subtitle = QtWidgets.QLabel("Plots update after a preview or FPGA run completes.")
        self.canvas = PlotCanvas(self)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.subtitle)
        layout.addWidget(self.canvas, 1)

    def update_result(self, result: RunResult) -> None:
        self.subtitle.setText(f"Displaying data from the {result.source} run at {result.timestamp}.")
        self.canvas.update_result(result)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Shack-Hartmann FPGA Control Shell")
        self.resize(1380, 920)

        self.service = FpgaControlService(self)

        self.tabs = QtWidgets.QTabWidget()
        self.results_tab = ResultsTab()
        self.accuracy_tab = AccuracyTab()
        self.comparison_tab = ComparisonTab()
        self.plots_tab = PlotsTab()

        self.tabs.addTab(self.results_tab, "Run + Results")
        self.tabs.addTab(self.accuracy_tab, "Accuracy Stats")
        self.tabs.addTab(self.comparison_tab, "HCIPy Comparison")
        self.tabs.addTab(self.plots_tab, "Plots")
        self.setCentralWidget(self.tabs)

        self.results_tab.start_server_requested.connect(self.service.start_server)
        self.results_tab.stop_server_requested.connect(self.service.stop_server)
        self.results_tab.preview_requested.connect(self.service.generate_local_preview)

        self.service.log_message.connect(self.results_tab.append_log)
        self.service.server_state_changed.connect(self.results_tab.set_server_state)
        self.service.client_state_changed.connect(self.results_tab.set_client_state)
        self.service.error_reported.connect(self._handle_error)
        self.service.run_completed.connect(self._handle_result)

        self.statusBar().showMessage("Ready")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.service.stop_server()
        super().closeEvent(event)

    def _handle_error(self, message: str) -> None:
        self.results_tab.append_log(message)
        self.statusBar().showMessage("Last action failed")
        QtWidgets.QMessageBox.critical(self, "FPGA Control Shell", message)

    def _handle_result(self, result: RunResult) -> None:
        self.results_tab.update_result(result)
        self.accuracy_tab.update_result(result)
        self.comparison_tab.update_result(result)
        self.plots_tab.update_result(result)
        self.statusBar().showMessage(f"Updated UI with the {result.source} run from {result.timestamp}")


def configure_qt_platform(
    *,
    platform_name: str | None = None,
    environment: MutableMapping[str, str] | None = None,
) -> None:
    platform_name = sys.platform if platform_name is None else platform_name
    environment = os.environ if environment is None else environment

    if environment.get("QT_QPA_PLATFORM"):
        return

    if platform_name.startswith("win") or platform_name == "darwin":
        return

    has_linux_display = bool(environment.get("DISPLAY") or environment.get("WAYLAND_DISPLAY"))
    if platform_name.startswith("linux") and not has_linux_display:
        raise RuntimeError(
            "No GUI display was detected. If you are launching from a dev container, WSL, or remote Linux shell, "
            "run the app with a local desktop Python on Windows, or explicitly set QT_QPA_PLATFORM=offscreen for tests."
        )


def build_application() -> QtWidgets.QApplication:
    configure_qt_platform()
    existing_app = QtWidgets.QApplication.instance()
    if existing_app is None:
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = cast(QtWidgets.QApplication, existing_app)
    app.setStyle("Fusion")
    app.setStyleSheet(
        """
        QMainWindow, QWidget {
            background: #f4efe6;
            color: #1d2a33;
        }
        QGroupBox {
            border: 1px solid #d5c8b5;
            border-radius: 10px;
            margin-top: 12px;
            padding-top: 12px;
            font-weight: 600;
            background: #fffaf2;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 4px 0 4px;
        }
        QPushButton {
            background: #264653;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 14px;
            min-height: 18px;
        }
        QPushButton:hover {
            background: #2f5d6c;
        }
        QPushButton:pressed {
            background: #1d3740;
        }
        QLineEdit, QSpinBox, QPlainTextEdit, QTableWidget, QTabWidget::pane {
            background: #fffdf8;
            border: 1px solid #d8ccba;
            border-radius: 8px;
        }
        QHeaderView::section {
            background: #e9dcc8;
            color: #1d2a33;
            padding: 6px;
            border: none;
        }
        QLabel#ValueCard {
            font-size: 18px;
            font-weight: 700;
            color: #b3532f;
        }
        QTabBar::tab {
            background: #dec9a6;
            color: #1d2a33;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            padding: 10px 16px;
            margin-right: 4px;
        }
        QTabBar::tab:selected {
            background: #264653;
            color: white;
        }
        """
    )
    return app


def main() -> int:
    app = build_application()
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())