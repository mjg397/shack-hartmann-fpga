from __future__ import annotations

import os
import sys
from collections.abc import MutableMapping
from typing import cast

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from fpga_gui_service import (
    DEFAULT_BIND_IP,
    DEFAULT_BIND_PORT,
    FPGA_RM_GLOBAL_SCALE,
    FpgaControlService,
    RunResult,
)


def _align_mode_labels_and_series(
    mode_labels: list[str], *series: np.ndarray | None
) -> tuple[list[str], list[np.ndarray | None]]:
    count = len(mode_labels)
    for values in series:
        if values is not None:
            count = min(count, len(values))

    aligned_labels = list(mode_labels[:count])
    aligned_series = [None if values is None else np.asarray(values)[:count] for values in series]
    return aligned_labels, aligned_series


class ResultsTab(QtWidgets.QWidget):
    start_server_requested = QtCore.Signal(str, int)
    stop_server_requested = QtCore.Signal()
    preview_requested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self.server_state_value = QtWidgets.QLabel("Stopped")
        self.client_state_value = QtWidgets.QLabel("No client connected")
        self.last_run_value = QtWidgets.QLabel("No runs yet")
        self.compute_time_value = QtWidgets.QLabel("n/a")
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
        status_layout.addWidget(self._make_card("FPGA Compute Time", self.compute_time_value), 0, 3)

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

        # Format FPGA compute time
        if result.fpga_compute_time_ns is not None:
            ns = result.fpga_compute_time_ns
            if ns < 1_000:
                time_text = f"{ns} ns"
            elif ns < 1_000_000:
                time_text = f"{ns / 1_000:.2f} µs ({ns:,} ns)"
            elif ns < 1_000_000_000:
                time_text = f"{ns / 1_000_000:.3f} ms ({ns:,} ns)"
            else:
                time_text = f"{ns / 1_000_000_000:.4f} s ({ns:,} ns)"
            self.compute_time_value.setText(time_text)
        else:
            self.compute_time_value.setText("n/a")

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
        mode_labels, aligned_series = _align_mode_labels_and_series(result.mode_labels, baseline, measured)
        baseline = cast(np.ndarray, aligned_series[0])
        measured = cast(np.ndarray, aligned_series[1])
        if not mode_labels:
            self.message_label.setText("No overlapping HCIPy/FPGA coefficient data is available for accuracy metrics.")
            self.metrics_table.setRowCount(0)
            for label in (self.rmse_value, self.mae_value, self.max_error_value, self.correlation_value):
                label.setText("n/a")
            return

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

        self.metrics_table.setRowCount(len(mode_labels))
        for row_index, mode_label in enumerate(mode_labels):
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
        mode_labels, aligned_series = _align_mode_labels_and_series(result.mode_labels, baseline_nm, measured_nm)
        baseline_nm = cast(np.ndarray, aligned_series[0])
        measured_nm = cast(np.ndarray, aligned_series[1])
        if not mode_labels:
            self.show_placeholder("No overlapping HCIPy/FPGA coefficient data is available yet.")
            return

        errors_nm = measured_nm - baseline_nm

        self.figure.clear()
        axes = self.figure.subplots(1, 2)

        combined = np.concatenate((baseline_nm, measured_nm))
        max_extent = max(1.0, float(np.max(np.abs(combined))) * 1.1)

        axes[0].scatter(baseline_nm, measured_nm, color="#e76f51", s=48)
        axes[0].plot([-max_extent, max_extent], [-max_extent, max_extent], linestyle="--", color="#264653")
        for index, label in enumerate(mode_labels):
            axes[0].annotate(label, (baseline_nm[index], measured_nm[index]), textcoords="offset points", xytext=(4, 4))
        axes[0].set_xlim(-max_extent, max_extent)
        axes[0].set_ylim(-max_extent, max_extent)
        axes[0].set_xlabel("HCIPy (nm)")
        axes[0].set_ylabel("FPGA (nm)")
        axes[0].set_title("Coefficient Correlation")
        axes[0].grid(alpha=0.25)

        x_axis = np.arange(len(mode_labels))
        colors = ["#e76f51" if value >= 0.0 else "#2a9d8f" for value in errors_nm]
        axes[1].bar(x_axis, errors_nm, color=colors)
        axes[1].axhline(0.0, color="#264653", linewidth=0.9)
        axes[1].set_xticks(x_axis)
        axes[1].set_xticklabels(mode_labels, rotation=45, ha="right")
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
        mode_labels, aligned_series = _align_mode_labels_and_series(result.mode_labels, baseline_nm, measured_nm)
        baseline_nm = cast(np.ndarray, aligned_series[0])
        measured_nm = cast(np.ndarray, aligned_series[1])
        if not mode_labels:
            self.message_label.setText("No overlapping HCIPy/FPGA coefficient data is available yet.")
            self.mean_bias_value.setText("n/a")
            self.worst_mode_value.setText("n/a")
            self.table.setRowCount(0)
            self.canvas.show_placeholder("No overlapping HCIPy/FPGA coefficient data is available yet.")
            return

        errors_nm = measured_nm - baseline_nm
        raw_values = result.fpga_zernikes_raw

        worst_index = int(np.argmax(np.abs(errors_nm)))
        self.message_label.setText(
            f"Comparing the HCIPy baseline against FPGA outputs from the run at {result.timestamp}."
        )
        self.mean_bias_value.setText(f"{float(np.mean(errors_nm)):+.3f} nm")
        self.worst_mode_value.setText(f"{mode_labels[worst_index]} ({errors_nm[worst_index]:+.3f} nm)")

        self.table.setRowCount(len(mode_labels))
        for row_index, mode_label in enumerate(mode_labels):
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


class ZernikeComparisonCanvas(FigureCanvasQTAgg):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        self.figure = Figure(figsize=(11, 6), tight_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)

    def show_placeholder(self, message: str) -> None:
        self.figure.clear()
        axis = self.figure.subplots()
        axis.text(0.5, 0.5, message, ha="center", va="center")
        axis.set_axis_off()
        self.draw_idle()

    def update_result(self, result: RunResult) -> None:
        true_nm = None if result.true_zernikes is None else np.asarray(result.true_zernikes, dtype=np.float64) * 1e9
        hcipy_nm = None if result.hcipy_zernikes is None else np.asarray(result.hcipy_zernikes, dtype=np.float64) * 1e9
        fpga_nm = None if result.fpga_zernikes is None else np.asarray(result.fpga_zernikes, dtype=np.float64) * 1e9

        if true_nm is None and hcipy_nm is None and fpga_nm is None:
            self.show_placeholder("No Zernike coefficient data is available yet.")
            return

        mode_labels, aligned_series = _align_mode_labels_and_series(result.mode_labels, true_nm, hcipy_nm, fpga_nm)
        true_nm = aligned_series[0]
        hcipy_nm = aligned_series[1]
        fpga_nm = aligned_series[2]
        if not mode_labels:
            self.show_placeholder("No Zernike coefficient data is available yet.")
            return

        self.figure.clear()
        axis = self.figure.subplots()

        x_axis = np.arange(len(mode_labels), dtype=np.float64)
        series_count = sum(values is not None for values in (true_nm, hcipy_nm, fpga_nm))
        bar_width = 0.22 if series_count >= 3 else 0.32
        center_offsets = {
            1: [0.0],
            2: [-bar_width / 2, bar_width / 2],
            3: [-bar_width, 0.0, bar_width],
        }[series_count]

        plotted_series: list[tuple[np.ndarray, str, str]] = []
        if true_nm is not None:
            plotted_series.append((true_nm, "True aberration", "#457b9d"))
        if hcipy_nm is not None:
            plotted_series.append((hcipy_nm, "HCIPy", "#2a9d8f"))
        if fpga_nm is not None:
            plotted_series.append((fpga_nm, "FPGA", "#e76f51"))

        for (values, label, color), offset in zip(plotted_series, center_offsets):
            axis.bar(x_axis + offset, values, width=bar_width, label=label, color=color)

        axis.axhline(0.0, color="#222222", linewidth=0.8)
        axis.set_xticks(x_axis)
        axis.set_xticklabels(mode_labels, rotation=45, ha="right")
        axis.set_ylabel("Coefficient [nm]")
        axis.set_title("Zernike coefficient comparison")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(loc="upper right")

        self.draw_idle()


class ZernikeComparisonTab(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self.message_label = QtWidgets.QLabel(
            "Compare the injected aberration against the HCIPy reconstruction and the FPGA coefficients when a hardware run completes."
        )
        self.message_label.setWordWrap(True)

        self.canvas = ZernikeComparisonCanvas(self)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Mode", "True (nm)", "HCIPy (nm)", "FPGA (nm)"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.message_label)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.table, 1)

        self.canvas.show_placeholder("Run a local preview or FPGA capture to populate the Zernike comparison chart.")

    def update_result(self, result: RunResult) -> None:
        if result.source == "local-preview":
            self.message_label.setText(
                "Displaying the injected aberration and HCIPy reconstruction from local preview. FPGA bars populate after a hardware run completes."
            )
        else:
            self.message_label.setText(
                f"Displaying True, HCIPy, and FPGA coefficients from the run at {result.timestamp}."
            )

        true_nm = None if result.true_zernikes is None else np.asarray(result.true_zernikes, dtype=np.float64) * 1e9
        hcipy_nm = None if result.hcipy_zernikes is None else np.asarray(result.hcipy_zernikes, dtype=np.float64) * 1e9
        fpga_nm = None if result.fpga_zernikes is None else np.asarray(result.fpga_zernikes, dtype=np.float64) * 1e9
        mode_labels, aligned_series = _align_mode_labels_and_series(result.mode_labels, true_nm, hcipy_nm, fpga_nm)
        true_nm = aligned_series[0]
        hcipy_nm = aligned_series[1]
        fpga_nm = aligned_series[2]

        self.table.setRowCount(len(mode_labels))
        for row_index, mode_label in enumerate(mode_labels):
            row_values = (
                mode_label,
                "n/a" if true_nm is None else f"{true_nm[row_index]:+.3f}",
                "n/a" if hcipy_nm is None else f"{hcipy_nm[row_index]:+.3f}",
                "n/a" if fpga_nm is None else f"{fpga_nm[row_index]:+.3f}",
            )
            for column, text in enumerate(row_values):
                item = QtWidgets.QTableWidgetItem(text)
                if column > 0:
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(row_index, column, item)

        self.canvas.update_result(result)


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        self.figure = Figure(figsize=(15, 11), tight_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)

    def show_placeholder(self, message: str) -> None:
        self.figure.clear()
        axis = self.figure.subplots()
        axis.text(0.5, 0.5, message, ha="center", va="center")
        axis.set_axis_off()
        self.draw_idle()

    def _reshape_square_image(self, image: np.ndarray | None) -> np.ndarray | None:
        if image is None:
            return None
        values = np.asarray(image, dtype=np.float64)
        if values.ndim == 2:
            return values
        if values.ndim != 1:
            return None
        side = int(round(np.sqrt(values.size)))
        if side * side != values.size:
            return None
        return values.reshape(side, side)

    def _masked_abs_max(self, image: np.ndarray | None, mask: np.ndarray | None = None) -> float:
        values = self._reshape_square_image(image)
        if values is None:
            return 0.0
        if mask is not None and mask.shape == values.shape:
            valid_values = values[np.asarray(mask, dtype=bool)]
        else:
            valid_values = values.ravel()
        if valid_values.size == 0:
            return 0.0
        return float(np.max(np.abs(valid_values)))

    def _masked_rms(self, image: np.ndarray | None, mask: np.ndarray | None = None) -> float | None:
        values = self._reshape_square_image(image)
        if values is None:
            return None
        if mask is not None and mask.shape == values.shape:
            valid_values = values[np.asarray(mask, dtype=bool)]
        else:
            valid_values = values.ravel()
        if valid_values.size == 0:
            return None
        return float(np.sqrt(np.mean(valid_values**2)))

    def _plot_image(
        self,
        axis: Axes,
        image: np.ndarray | None,
        title: str,
        *,
        cmap: str = "inferno",
        colorbar_label: str | None = None,
        symmetric: bool = False,
        mask: np.ndarray | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        values = self._reshape_square_image(image)
        if values is None:
            axis.text(0.5, 0.5, "No data", ha="center", va="center")
            axis.set_title(title)
            axis.set_xticks([])
            axis.set_yticks([])
            return

        plot_values: np.ndarray | np.ma.MaskedArray
        plot_values = values
        if mask is not None and mask.shape == values.shape:
            plot_values = np.ma.array(values, mask=~np.asarray(mask, dtype=bool))

        if symmetric and vmin is None and vmax is None:
            valid_values = np.asarray(np.ma.array(plot_values).compressed(), dtype=np.float64)
            dynamic_range = float(np.max(np.abs(valid_values))) if valid_values.size else 1.0
            vmin = -dynamic_range
            vmax = dynamic_range

        image_artist = axis.imshow(plot_values, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        if colorbar_label is not None:
            colorbar = self.figure.colorbar(image_artist, ax=axis, fraction=0.046, pad=0.04)
            colorbar.set_label(colorbar_label)

    def _plot_slope_field(
        self,
        axis: Axes,
        slopes_xy_grid: np.ndarray | None,
        mask: np.ndarray | None,
        title: str,
    ) -> None:
        if slopes_xy_grid is None:
            axis.text(0.5, 0.5, "No slope data", ha="center", va="center")
            axis.set_title(title)
            axis.set_xticks([])
            axis.set_yticks([])
            return

        slopes = np.asarray(slopes_xy_grid, dtype=np.float64)
        local_mask = np.ones(slopes.shape[:2], dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        slope_x = np.ma.array(slopes[..., 0], mask=~local_mask)
        slope_y = np.ma.array(slopes[..., 1], mask=~local_mask)
        magnitude = np.ma.sqrt(slope_x**2 + slope_y**2)
        image_artist = axis.imshow(magnitude, cmap="viridis", origin="lower")

        row_indices, col_indices = np.indices(local_mask.shape)
        valid_rows = row_indices[local_mask]
        valid_cols = col_indices[local_mask]
        valid_x = np.asarray(slope_x[local_mask], dtype=np.float64)
        valid_y = np.asarray(slope_y[local_mask], dtype=np.float64)
        if valid_x.size:
            max_magnitude = float(np.max(np.hypot(valid_x, valid_y)))
            quiver_scale = max(max_magnitude * 3.0, 1e-6)
            axis.quiver(
                valid_cols,
                valid_rows,
                valid_x,
                valid_y,
                color="white",
                angles="xy",
                scale_units="xy",
                scale=quiver_scale,
                width=0.007,
            )

        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        colorbar = self.figure.colorbar(image_artist, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label("slope magnitude [px]")

    def _plot_slope_delta(
        self,
        axis: Axes,
        fpga_slopes_xy_grid: np.ndarray | None,
        hcipy_slopes_xy_grid: np.ndarray | None,
        mask: np.ndarray | None,
        title: str,
    ) -> None:
        if fpga_slopes_xy_grid is None or hcipy_slopes_xy_grid is None:
            axis.text(0.5, 0.5, "Need HCIPy and FPGA slopes", ha="center", va="center")
            axis.set_title(title)
            axis.set_xticks([])
            axis.set_yticks([])
            return

        fpga_slopes = np.asarray(fpga_slopes_xy_grid, dtype=np.float64)
        hcipy_slopes = np.asarray(hcipy_slopes_xy_grid, dtype=np.float64)
        delta = fpga_slopes - hcipy_slopes
        magnitude = np.linalg.norm(delta, axis=2)
        if mask is not None:
            magnitude = np.ma.array(magnitude, mask=~np.asarray(mask, dtype=bool))

        image_artist = axis.imshow(magnitude, cmap="magma", origin="lower")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        colorbar = self.figure.colorbar(image_artist, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label("|FPGA - HCIPy| [px]")

    def _plot_coefficients(self, axis: Axes, result: RunResult) -> None:
        axis.axhline(0.0, color="#222222", linewidth=0.8)

        true_nm = None if result.true_zernikes is None else np.asarray(result.true_zernikes, dtype=np.float64) * 1e9
        hcipy_nm = None if result.hcipy_zernikes is None else np.asarray(result.hcipy_zernikes, dtype=np.float64) * 1e9
        fpga_nm = None if result.fpga_zernikes is None else np.asarray(result.fpga_zernikes, dtype=np.float64) * 1e9

        if true_nm is None and hcipy_nm is None and fpga_nm is None:
            axis.text(0.5, 0.5, "No coefficient data", ha="center", va="center")
            axis.set_title("Zernike coefficient comparison")
            axis.set_xticks([])
            axis.set_yticks([])
            return

        mode_labels, aligned_series = _align_mode_labels_and_series(result.mode_labels, true_nm, hcipy_nm, fpga_nm)
        true_nm = aligned_series[0]
        hcipy_nm = aligned_series[1]
        fpga_nm = aligned_series[2]
        if not mode_labels:
            axis.text(0.5, 0.5, "No coefficient data", ha="center", va="center")
            axis.set_title("Zernike coefficient comparison")
            axis.set_xticks([])
            axis.set_yticks([])
            return

        x_axis = np.arange(len(mode_labels))

        if true_nm is not None:
            axis.plot(x_axis, true_nm, marker="o", linewidth=1.8, color="#457b9d", label="True")
        if hcipy_nm is not None:
            axis.plot(x_axis, hcipy_nm, marker="s", linewidth=1.8, color="#2a9d8f", label="HCIPy")
        if fpga_nm is not None:
            axis.plot(x_axis, fpga_nm, marker="^", linewidth=1.8, color="#e76f51", label="FPGA")

        axis.set_xticks(x_axis)
        axis.set_xticklabels(mode_labels, rotation=45, ha="right")
        axis.set_ylabel("Coefficient [nm]")
        axis.set_title("Zernike coefficient comparison")
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right")

    def _compute_log_psf(self, opd_map_nm: np.ndarray | None, aperture_mask: np.ndarray | None, wavelength_m: float | None) -> np.ndarray | None:
        if opd_map_nm is None or aperture_mask is None or wavelength_m is None:
            return None
        opd_m = np.asarray(opd_map_nm, dtype=np.float64) * 1e-9
        pupil = np.asarray(aperture_mask, dtype=np.float64) * np.exp(1j * opd_m * (2.0 * np.pi / wavelength_m))
        focal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil)))
        psf = np.abs(focal) ** 2
        peak = float(np.max(psf))
        if peak <= 0.0:
            return None
        return np.log10(np.clip(psf / peak, 1e-12, None))

    def update_result(self, result: RunResult) -> None:
        self.figure.clear()
        axes = self.figure.subplots(3, 4)

        is_local_preview = result.source == "local-preview"
        fpga_label = "FPGA-like (host emulation)" if is_local_preview else "FPGA"
        frame_title = "Quantized detector frame for FPGA" if is_local_preview else "Detector frame sent to FPGA"
        slope_delta_title = f"{fpga_label} vs HCIPy slope mismatch"
        opd_vmax = max(
            1.0,
            self._masked_abs_max(result.input_opd_map_nm, result.aperture_mask),
            self._masked_abs_max(result.reconstructed_opd_map_nm, result.aperture_mask),
            self._masked_abs_max(result.residual_opd_map_nm, result.aperture_mask),
        )
        residual_rms_nm = self._masked_rms(result.residual_opd_map_nm, result.aperture_mask)
        residual_title = "Residual OPD"
        if residual_rms_nm is not None:
            residual_title = f"Residual OPD (RMS {residual_rms_nm:.1f} nm)"

        self.figure.suptitle(
            f"HCIPy / {fpga_label} diagnostics for {result.source} run at {result.timestamp}",
            fontsize=14,
            fontweight="bold",
        )

        self._plot_image(
            axes[0, 0],
            result.reference_image,
            "Reference WFS image",
            cmap="inferno",
            colorbar_label="counts",
        )
        self._plot_image(
            axes[0, 1],
            result.quantized_image,
            frame_title,
            cmap="inferno",
            colorbar_label="ADU",
        )
        self._plot_slope_field(axes[0, 2], result.hcipy_slopes_xy_grid, result.valid_subaperture_mask, "HCIPy slope field")
        self._plot_slope_field(axes[0, 3], result.slopes_xy_grid, result.valid_subaperture_mask, f"{fpga_label} slope field")

        self._plot_image(
            axes[1, 0],
            result.input_opd_map_nm,
            "Input OPD",
            cmap="RdBu_r",
            colorbar_label="nm",
            symmetric=True,
            mask=result.aperture_mask,
            vmin=-opd_vmax,
            vmax=opd_vmax,
        )
        self._plot_image(
            axes[1, 1],
            result.reconstructed_opd_map_nm,
            "Reconstructed OPD",
            cmap="RdBu_r",
            colorbar_label="nm",
            symmetric=True,
            mask=result.aperture_mask,
            vmin=-opd_vmax,
            vmax=opd_vmax,
        )
        self._plot_image(
            axes[1, 2],
            result.residual_opd_map_nm,
            residual_title,
            cmap="RdBu_r",
            colorbar_label="nm",
            symmetric=True,
            mask=result.aperture_mask,
            vmin=-opd_vmax,
            vmax=opd_vmax,
        )
        self._plot_slope_delta(
            axes[1, 3],
            result.slopes_xy_grid,
            result.hcipy_slopes_xy_grid,
            result.valid_subaperture_mask,
            slope_delta_title,
        )

        diffraction_limited_psf = self._compute_log_psf(
            np.zeros_like(result.input_opd_map_nm) if result.input_opd_map_nm is not None else None,
            result.aperture_mask,
            result.wavelength_m,
        )
        aberrated_psf = self._compute_log_psf(result.input_opd_map_nm, result.aperture_mask, result.wavelength_m)
        reconstructed_psf = self._compute_log_psf(result.residual_opd_map_nm, result.aperture_mask, result.wavelength_m)

        self._plot_image(
            axes[2, 0],
            diffraction_limited_psf,
            "Diffraction-limited PSF",
            cmap="inferno",
            colorbar_label="log10(I / Imax)",
            vmin=-6,
            vmax=0,
        )
        self._plot_image(
            axes[2, 1],
            aberrated_psf,
            "Aberrated PSF",
            cmap="inferno",
            colorbar_label="log10(I / Imax)",
            vmin=-6,
            vmax=0,
        )
        self._plot_image(
            axes[2, 2],
            reconstructed_psf,
            "Reconstructed PSF",
            cmap="inferno",
            colorbar_label="log10(I / Imax)",
            vmin=-6,
            vmax=0,
        )
        self._plot_coefficients(axes[2, 3], result)

        self.draw_idle()


class PlotsTab(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.subtitle = QtWidgets.QLabel(
            "HCIPy-style diagnostics update after a preview or FPGA run completes. Use the toolbar to zoom into pupil, WFS, and PSF structure."
        )
        self.subtitle.setWordWrap(True)
        self.canvas = PlotCanvas(self)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.subtitle)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

    def update_result(self, result: RunResult) -> None:
        preview_note = " For local preview, FPGA-labeled panels are host-side fixed-point emulation, not hardware readback." if result.source == "local-preview" else ""
        self.subtitle.setText(
            f"Displaying HCIPy-style diagnostics from the {result.source} run at {result.timestamp}. The dashboard aligns WFS imagery, slope fields, OPD maps, PSFs, and coefficient traces in one view.{preview_note}"
        )
        self.canvas.update_result(result)


class FpgaPlotCanvas(FigureCanvasQTAgg):
    """Diagnostic plot canvas using FPGA-reconstructed OPD maps.

    This mirrors PlotCanvas but substitutes the HCIPy OPD reconstruction
    with the OPD reconstructed from the FPGA Zernike coefficients, giving
    a side-by-side visual comparison of FPGA reconstruction quality.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        self.figure = Figure(figsize=(15, 11), tight_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)

    def show_placeholder(self, message: str) -> None:
        self.figure.clear()
        axis = self.figure.subplots()
        axis.text(0.5, 0.5, message, ha="center", va="center")
        axis.set_axis_off()
        self.draw_idle()

    def _reshape_square_image(self, image: np.ndarray | None) -> np.ndarray | None:
        if image is None:
            return None
        values = np.asarray(image, dtype=np.float64)
        if values.ndim == 2:
            return values
        if values.ndim != 1:
            return None
        side = int(round(np.sqrt(values.size)))
        if side * side != values.size:
            return None
        return values.reshape(side, side)

    def _masked_abs_max(self, image: np.ndarray | None, mask: np.ndarray | None = None) -> float:
        values = self._reshape_square_image(image)
        if values is None:
            return 0.0
        if mask is not None and mask.shape == values.shape:
            valid_values = values[np.asarray(mask, dtype=bool)]
        else:
            valid_values = values.ravel()
        if valid_values.size == 0:
            return 0.0
        return float(np.max(np.abs(valid_values)))

    def _masked_rms(self, image: np.ndarray | None, mask: np.ndarray | None = None) -> float | None:
        values = self._reshape_square_image(image)
        if values is None:
            return None
        if mask is not None and mask.shape == values.shape:
            valid_values = values[np.asarray(mask, dtype=bool)]
        else:
            valid_values = values.ravel()
        if valid_values.size == 0:
            return None
        return float(np.sqrt(np.mean(valid_values**2)))

    def _plot_image(
        self,
        axis: Axes,
        image: np.ndarray | None,
        title: str,
        *,
        cmap: str = "inferno",
        colorbar_label: str | None = None,
        symmetric: bool = False,
        mask: np.ndarray | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        values = self._reshape_square_image(image)
        if values is None:
            axis.text(0.5, 0.5, "No data", ha="center", va="center")
            axis.set_title(title)
            axis.set_xticks([])
            axis.set_yticks([])
            return

        plot_values: np.ndarray | np.ma.MaskedArray
        plot_values = values
        if mask is not None and mask.shape == values.shape:
            plot_values = np.ma.array(values, mask=~np.asarray(mask, dtype=bool))

        if symmetric and vmin is None and vmax is None:
            valid_values = np.asarray(np.ma.array(plot_values).compressed(), dtype=np.float64)
            dynamic_range = float(np.max(np.abs(valid_values))) if valid_values.size else 1.0
            vmin = -dynamic_range
            vmax = dynamic_range

        image_artist = axis.imshow(plot_values, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        if colorbar_label is not None:
            colorbar = self.figure.colorbar(image_artist, ax=axis, fraction=0.046, pad=0.04)
            colorbar.set_label(colorbar_label)

    def _plot_slope_field(
        self,
        axis: Axes,
        slopes_xy_grid: np.ndarray | None,
        mask: np.ndarray | None,
        title: str,
    ) -> None:
        if slopes_xy_grid is None:
            axis.text(0.5, 0.5, "No slope data", ha="center", va="center")
            axis.set_title(title)
            axis.set_xticks([])
            axis.set_yticks([])
            return

        slopes = np.asarray(slopes_xy_grid, dtype=np.float64)
        local_mask = np.ones(slopes.shape[:2], dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        slope_x = np.ma.array(slopes[..., 0], mask=~local_mask)
        slope_y = np.ma.array(slopes[..., 1], mask=~local_mask)
        magnitude = np.ma.sqrt(slope_x**2 + slope_y**2)
        image_artist = axis.imshow(magnitude, cmap="viridis", origin="lower")

        row_indices, col_indices = np.indices(local_mask.shape)
        valid_rows = row_indices[local_mask]
        valid_cols = col_indices[local_mask]
        valid_x = np.asarray(slope_x[local_mask], dtype=np.float64)
        valid_y = np.asarray(slope_y[local_mask], dtype=np.float64)
        if valid_x.size:
            max_magnitude = float(np.max(np.hypot(valid_x, valid_y)))
            quiver_scale = max(max_magnitude * 3.0, 1e-6)
            axis.quiver(
                valid_cols,
                valid_rows,
                valid_x,
                valid_y,
                color="white",
                angles="xy",
                scale_units="xy",
                scale=quiver_scale,
                width=0.007,
            )

        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        colorbar = self.figure.colorbar(image_artist, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label("slope magnitude [px]")

    def _plot_coefficients(self, axis: Axes, result: RunResult) -> None:
        axis.axhline(0.0, color="#222222", linewidth=0.8)

        true_nm = None if result.true_zernikes is None else np.asarray(result.true_zernikes, dtype=np.float64) * 1e9
        fpga_nm = None if result.fpga_zernikes is None else np.asarray(result.fpga_zernikes, dtype=np.float64) * 1e9

        if true_nm is None and fpga_nm is None:
            axis.text(0.5, 0.5, "No coefficient data", ha="center", va="center")
            axis.set_title("FPGA Zernike coefficients")
            axis.set_xticks([])
            axis.set_yticks([])
            return

        mode_labels, aligned_series = _align_mode_labels_and_series(result.mode_labels, true_nm, fpga_nm)
        true_nm = aligned_series[0]
        fpga_nm = aligned_series[1]
        if not mode_labels:
            axis.text(0.5, 0.5, "No coefficient data", ha="center", va="center")
            axis.set_title("FPGA Zernike coefficients")
            axis.set_xticks([])
            axis.set_yticks([])
            return

        x_axis = np.arange(len(mode_labels))

        if true_nm is not None:
            axis.plot(x_axis, true_nm, marker="o", linewidth=1.8, color="#457b9d", label="True")
        if fpga_nm is not None:
            axis.plot(x_axis, fpga_nm, marker="^", linewidth=1.8, color="#e76f51", label="FPGA")

        axis.set_xticks(x_axis)
        axis.set_xticklabels(mode_labels, rotation=45, ha="right")
        axis.set_ylabel("Coefficient [nm]")
        axis.set_title("FPGA Zernike coefficients")
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right")

    def _compute_log_psf(self, opd_map_nm: np.ndarray | None, aperture_mask: np.ndarray | None, wavelength_m: float | None) -> np.ndarray | None:
        if opd_map_nm is None or aperture_mask is None or wavelength_m is None:
            return None
        opd_m = np.asarray(opd_map_nm, dtype=np.float64) * 1e-9
        pupil = np.asarray(aperture_mask, dtype=np.float64) * np.exp(1j * opd_m * (2.0 * np.pi / wavelength_m))
        focal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil)))
        psf = np.abs(focal) ** 2
        peak = float(np.max(psf))
        if peak <= 0.0:
            return None
        return np.log10(np.clip(psf / peak, 1e-12, None))

    def update_result(self, result: RunResult) -> None:
        fpga_recon = result.fpga_reconstructed_opd_map_nm
        fpga_resid = result.fpga_residual_opd_map_nm

        if fpga_recon is None and fpga_resid is None:
            self.show_placeholder(
                "FPGA reconstruction plots appear after an FPGA hardware run completes.\n"
                "The FPGA Zernike coefficients are used to reconstruct the OPD map."
            )
            return

        self.figure.clear()
        axes = self.figure.subplots(3, 4)

        self.figure.suptitle(
            f"FPGA reconstruction diagnostics for {result.source} run at {result.timestamp}",
            fontsize=14,
            fontweight="bold",
        )

        # Row 0: WFS images and FPGA slope field
        self._plot_image(
            axes[0, 0],
            result.reference_image,
            "Reference WFS image",
            cmap="inferno",
            colorbar_label="counts",
        )
        self._plot_image(
            axes[0, 1],
            result.quantized_image,
            "Detector frame sent to FPGA",
            cmap="inferno",
            colorbar_label="ADU",
        )
        self._plot_slope_field(axes[0, 2], result.slopes_xy_grid, result.valid_subaperture_mask, "FPGA slope field")

        # Row 0, col 3: empty placeholder for symmetry
        axes[0, 3].set_axis_off()

        # Row 1: OPD maps — Input, FPGA Reconstructed, FPGA Residual
        opd_vmax = max(
            1.0,
            self._masked_abs_max(result.input_opd_map_nm, result.aperture_mask),
            self._masked_abs_max(fpga_recon, result.aperture_mask),
            self._masked_abs_max(fpga_resid, result.aperture_mask),
        )
        residual_rms_nm = self._masked_rms(fpga_resid, result.aperture_mask)
        fpga_residual_title = "FPGA Residual OPD"
        if residual_rms_nm is not None:
            fpga_residual_title = f"FPGA Residual OPD (RMS {residual_rms_nm:.1f} nm)"

        self._plot_image(
            axes[1, 0],
            result.input_opd_map_nm,
            "Input OPD",
            cmap="RdBu_r",
            colorbar_label="nm",
            symmetric=True,
            mask=result.aperture_mask,
            vmin=-opd_vmax,
            vmax=opd_vmax,
        )
        self._plot_image(
            axes[1, 1],
            fpga_recon,
            "FPGA Reconstructed OPD",
            cmap="RdBu_r",
            colorbar_label="nm",
            symmetric=True,
            mask=result.aperture_mask,
            vmin=-opd_vmax,
            vmax=opd_vmax,
        )
        self._plot_image(
            axes[1, 2],
            fpga_resid,
            fpga_residual_title,
            cmap="RdBu_r",
            colorbar_label="nm",
            symmetric=True,
            mask=result.aperture_mask,
            vmin=-opd_vmax,
            vmax=opd_vmax,
        )

        # Row 1, col 3: HCIPy vs FPGA residual difference
        hcipy_resid = result.residual_opd_map_nm
        if hcipy_resid is not None and fpga_resid is not None:
            hcipy_2d = self._reshape_square_image(hcipy_resid)
            fpga_2d = self._reshape_square_image(fpga_resid)
            if hcipy_2d is not None and fpga_2d is not None and hcipy_2d.shape == fpga_2d.shape:
                delta = fpga_resid - hcipy_resid
                self._plot_image(
                    axes[1, 3],
                    delta,
                    "FPGA − HCIPy Residual",
                    cmap="RdBu_r",
                    colorbar_label="nm",
                    symmetric=True,
                    mask=result.aperture_mask,
                )
            else:
                axes[1, 3].set_axis_off()
        else:
            axes[1, 3].set_axis_off()

        # Row 2: PSFs and coefficient trace
        diffraction_limited_psf = self._compute_log_psf(
            np.zeros_like(result.input_opd_map_nm) if result.input_opd_map_nm is not None else None,
            result.aperture_mask,
            result.wavelength_m,
        )
        aberrated_psf = self._compute_log_psf(result.input_opd_map_nm, result.aperture_mask, result.wavelength_m)
        fpga_corrected_psf = self._compute_log_psf(fpga_resid, result.aperture_mask, result.wavelength_m)

        self._plot_image(
            axes[2, 0],
            diffraction_limited_psf,
            "Diffraction-limited PSF",
            cmap="inferno",
            colorbar_label="log10(I / Imax)",
            vmin=-6,
            vmax=0,
        )
        self._plot_image(
            axes[2, 1],
            aberrated_psf,
            "Aberrated PSF",
            cmap="inferno",
            colorbar_label="log10(I / Imax)",
            vmin=-6,
            vmax=0,
        )
        self._plot_image(
            axes[2, 2],
            fpga_corrected_psf,
            "FPGA-corrected PSF",
            cmap="inferno",
            colorbar_label="log10(I / Imax)",
            vmin=-6,
            vmax=0,
        )
        self._plot_coefficients(axes[2, 3], result)

        self.draw_idle()


class FpgaPlotsTab(QtWidgets.QWidget):
    """Diagnostics tab using FPGA-reconstructed OPD maps instead of HCIPy."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.subtitle = QtWidgets.QLabel(
            "FPGA reconstruction diagnostics appear after a hardware FPGA run completes. "
            "OPD maps and PSFs are reconstructed from the FPGA Zernike coefficients."
        )
        self.subtitle.setWordWrap(True)
        self.canvas = FpgaPlotCanvas(self)
        self.canvas.show_placeholder(
            "FPGA reconstruction plots appear after an FPGA hardware run completes."
        )
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.subtitle)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

    def update_result(self, result: RunResult) -> None:
        if result.fpga_reconstructed_opd_map_nm is not None:
            self.subtitle.setText(
                f"Displaying FPGA reconstruction diagnostics from the {result.source} run at {result.timestamp}. "
                f"OPD maps and PSFs are reconstructed from the FPGA Zernike coefficients."
            )
        elif result.source == "local-preview":
            self.subtitle.setText(
                "Local preview does not produce FPGA Zernike coefficients. "
                "Run an FPGA hardware capture to populate this tab."
            )
        self.canvas.update_result(result)


class AberrationConfigTab(QtWidgets.QWidget):
    """Interactive Zernike aberration configuration for live demos."""

    coefficients_changed = QtCore.Signal(object)  # np.ndarray | None

    WAVELENGTH = 0.7e-6  # metres — must match shwfs_utils
    MODE_LABELS = [
        "Tilt X",
        "Tilt Y",
        "Defocus",
        "Astig 45\u00b0",
        "Astig 0\u00b0",
        "Coma X",
        "Coma Y",
        "Trefoil X",
        "Trefoil Y",
        "Spherical",
    ]
    SLIDER_RESOLUTION = 1000  # slider ticks per 1.0 lambda
    SLIDER_RANGE_LAMBDA = 0.030  # ±0.030 lambda (limited by 27-bit Q4.22 output in ematrix_accumulator)

    PRESETS: dict[str, list[float]] = {
        "Defaults": [0.010, 0.007, 0.008, 0.005, -0.004, 0.0, 0.0, 0.0, 0.0, 0.0],
        "Zero (flat)": [0.0] * 10,
        "Pure Defocus": [0.0, 0.0, 0.020, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "Pure Astigmatism": [0.0, 0.0, 0.0, 0.015, 0.010, 0.0, 0.0, 0.0, 0.0, 0.0],
        "Pure Coma": [0.0, 0.0, 0.0, 0.0, 0.0, 0.015, 0.010, 0.0, 0.0, 0.0],
        "Pure Spherical": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.020],
        "Pure Trefoil": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.015, 0.010, 0.0],
        "Heavy Mix": [0.015, 0.012, 0.020, 0.010, -0.008, 0.006, -0.005, 0.004, 0.003, 0.010],
        "Tilt Only": [0.020, 0.015, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self._sliders: list[QtWidgets.QSlider] = []
        self._value_labels: list[QtWidgets.QLabel] = []
        self._suppressing_signals = False

        # ── header ──
        header = QtWidgets.QLabel(
            "Configure the Zernike aberration coefficients injected into the HCIPy "
            "simulation. Values are in fractions of \u03bb (\u03bb = 700 nm). "
            "Changes take effect on the next Run Local Preview or FPGA run."
        )
        header.setWordWrap(True)

        # ── presets ──
        preset_box = QtWidgets.QGroupBox("Presets")
        preset_layout = QtWidgets.QHBoxLayout(preset_box)
        for preset_name in self.PRESETS:
            btn = QtWidgets.QPushButton(preset_name)
            btn.clicked.connect(lambda _checked=False, name=preset_name: self._apply_preset(name))
            preset_layout.addWidget(btn)
        preset_layout.addStretch()

        # ── sliders ──
        slider_box = QtWidgets.QGroupBox("Zernike Coefficients")
        slider_grid = QtWidgets.QGridLayout(slider_box)
        slider_grid.setColumnStretch(1, 1)

        for index, label in enumerate(self.MODE_LABELS):
            noll_index = index + 2
            mode_label = QtWidgets.QLabel(f"Z{noll_index}  {label}")
            mode_label.setMinimumWidth(120)

            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            tick_range = int(self.SLIDER_RANGE_LAMBDA * self.SLIDER_RESOLUTION)
            slider.setRange(-tick_range, tick_range)
            slider.setValue(0)
            slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
            slider.setTickInterval(self.SLIDER_RESOLUTION // 10)
            slider.valueChanged.connect(self._on_slider_changed)

            value_label = QtWidgets.QLabel("0.000 \u03bb")
            value_label.setMinimumWidth(90)
            value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

            slider_grid.addWidget(mode_label, index, 0)
            slider_grid.addWidget(slider, index, 1)
            slider_grid.addWidget(value_label, index, 2)

            self._sliders.append(slider)
            self._value_labels.append(value_label)

        # ── summary ──
        self._summary_label = QtWidgets.QLabel("")
        self._summary_label.setWordWrap(True)
        self._summary_label.setObjectName("ValueCard")

        # ── layout ──
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(header)
        layout.addWidget(preset_box)
        layout.addWidget(slider_box, 1)
        layout.addWidget(self._summary_label)

        # Apply defaults
        self._apply_preset("Defaults")

    # ── internal ──

    def _slider_to_lambda(self, tick_value: int) -> float:
        return tick_value / self.SLIDER_RESOLUTION

    def _lambda_to_slider(self, lam_fraction: float) -> int:
        return int(round(lam_fraction * self.SLIDER_RESOLUTION))

    def _on_slider_changed(self) -> None:
        if self._suppressing_signals:
            return
        for index, slider in enumerate(self._sliders):
            lam = self._slider_to_lambda(slider.value())
            self._value_labels[index].setText(f"{lam:+.3f} \u03bb")
        self._update_summary()
        self._emit_coefficients()

    def _apply_preset(self, name: str) -> None:
        values = self.PRESETS[name]
        self._suppressing_signals = True
        for index, lam in enumerate(values):
            self._sliders[index].setValue(self._lambda_to_slider(lam))
            self._value_labels[index].setText(f"{lam:+.3f} \u03bb")
        self._suppressing_signals = False
        self._update_summary()
        self._emit_coefficients()

    def _update_summary(self) -> None:
        parts: list[str] = []
        for index, label in enumerate(self.MODE_LABELS):
            lam = self._slider_to_lambda(self._sliders[index].value())
            if abs(lam) > 0.001:
                parts.append(f"{label}: {lam:+.3f}\u03bb")
        if parts:
            self._summary_label.setText("Active: " + ", ".join(parts))
        else:
            self._summary_label.setText("All coefficients are zero (flat wavefront).")

    def _emit_coefficients(self) -> None:
        coeffs_lambda = np.array(
            [self._slider_to_lambda(s.value()) for s in self._sliders],
            dtype=np.float64,
        )
        coeffs_metres = coeffs_lambda * self.WAVELENGTH
        self.coefficients_changed.emit(coeffs_metres)

    def get_coefficients_metres(self) -> np.ndarray:
        return np.array(
            [self._slider_to_lambda(s.value()) * self.WAVELENGTH for s in self._sliders],
            dtype=np.float64,
        )


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Shack-Hartmann FPGA Control Shell")
        self.resize(1380, 920)

        self.service = FpgaControlService(self)

        self.tabs = QtWidgets.QTabWidget()
        self.config_tab = AberrationConfigTab()
        self.results_tab = ResultsTab()
        self.accuracy_tab = AccuracyTab()
        self.comparison_tab = ComparisonTab()
        self.zernike_tab = ZernikeComparisonTab()
        self.plots_tab = PlotsTab()
        self.fpga_plots_tab = FpgaPlotsTab()

        self.tabs.addTab(self.config_tab, "\u2699 Aberration Config")
        self.tabs.addTab(self.results_tab, "Run + Results")
        self.tabs.addTab(self.accuracy_tab, "Accuracy Stats")
        self.tabs.addTab(self.comparison_tab, "HCIPy Comparison")
        self.tabs.addTab(self.zernike_tab, "Zernike Chart")
        self.tabs.addTab(self.plots_tab, "Plots (HCIPy)")
        self.tabs.addTab(self.fpga_plots_tab, "Plots (FPGA)")
        self.setCentralWidget(self.tabs)

        self.results_tab.start_server_requested.connect(self.service.start_server)
        self.results_tab.stop_server_requested.connect(self.service.stop_server)
        self.results_tab.preview_requested.connect(self.service.generate_local_preview)

        self.config_tab.coefficients_changed.connect(self.service.set_true_coeffs)

        self.service.log_message.connect(self.results_tab.append_log)
        self.service.server_state_changed.connect(self.results_tab.set_server_state)
        self.service.client_state_changed.connect(self.results_tab.set_client_state)
        self.service.error_reported.connect(self._handle_error)
        self.service.run_completed.connect(self._handle_result)

        # Push the initial default coefficients to the service
        self.service.set_true_coeffs(self.config_tab.get_coefficients_metres())

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
        self.zernike_tab.update_result(result)
        self.plots_tab.update_result(result)
        self.fpga_plots_tab.update_result(result)
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