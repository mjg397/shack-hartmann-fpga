from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import socket
import threading
import traceback

import numpy as np
from PySide6 import QtCore

from shwfs_utils import (
    generate_shwfs_case,
    quantize_shwfs_image,
    reshape_subaperture_xy,
    run_fpga_like_estimation,
    run_hcipy_estimation,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RESULT_DUMP_DIR = REPO_ROOT / "output_dumps"

CENTROID_SIZE = 4
SLOPE_SIZE = 4
ZERNIKE_SIZE = 4
CLIENT_TIMEOUT_SEC = 5.0

NUM_SUBAPERTURES_SIDE = 16
NUM_TOTAL_SUBAPERTURES = NUM_SUBAPERTURES_SIDE * NUM_SUBAPERTURES_SIDE # 256
NUM_VALID_SUBAPERTURES = 168 # The new standard
SUBAPERTURE_PIXELS = 16
NUM_PACKED_XY = NUM_VALID_SUBAPERTURES * 2
NUM_ZERNIKE = 10
Q4_23_SCALE = float(1 << 23)

DEFAULT_BIND_IP = "10.48.69.89"
DEFAULT_BIND_PORT = 80


@dataclass(slots=True)
class RunResult:
    source: str
    timestamp: str
    quantized_image: np.ndarray
    centroids_xy_grid: np.ndarray | None
    slopes_xy_grid: np.ndarray | None
    fpga_zernikes: np.ndarray | None
    hcipy_zernikes: np.ndarray | None
    mode_labels: list[str]
    notes: list[str] = field(default_factory=list)
    client_address: str | None = None
    simulation_data: dict | None = None
    hcipy_estimation_data: dict | None = None
    true_coeffs: np.ndarray | None = None

    @property
    def has_accuracy_comparison(self) -> bool:
        return self.fpga_zernikes is not None and self.hcipy_zernikes is not None

    @property
    def coefficient_errors(self) -> np.ndarray | None:
        if not self.has_accuracy_comparison:
            return None
        return np.asarray(self.fpga_zernikes, dtype=np.float64) - np.asarray(self.hcipy_zernikes, dtype=np.float64)


def recv_exact(sock: socket.socket, size: int) -> bytes | None:
    data = bytearray()
    while len(data) < size:
        try:
            chunk = sock.recv(size - len(data))
        except socket.timeout:
            return None
        except OSError:
            return None
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def recv_line(sock: socket.socket) -> bytes | None:
    data = bytearray()
    while True:
        try:
            chunk = sock.recv(1)
        except socket.timeout:
            return None
        except OSError:
            return None
        if not chunk:
            return None
        if chunk == b"\n":
            return bytes(data)
        data.extend(chunk)


def wait_for_exact_line(sock: socket.socket, expected: bytes, max_lines: int = 8) -> bool:
    for _ in range(max_lines):
        line = recv_line(sock)
        if line is None:
            return False
        if line == expected:
            return True
    return False


def decode_q4_23_from_bytes(payload: bytes, expected_count: int) -> np.ndarray:
    values_i32 = np.frombuffer(payload, dtype="<i4", count=expected_count)
    return values_i32.astype(np.float64) / Q4_23_SCALE


def unpack_xy_vectors(packed_values: np.ndarray, n_vectors: int) -> np.ndarray:
    if packed_values.size != 2 * n_vectors:
        raise ValueError(f"Expected {2 * n_vectors} packed XY values, got {packed_values.size}")
    x_vals = packed_values[:n_vectors]
    y_vals = packed_values[n_vectors:]
    return np.column_stack((x_vals, y_vals))


def save_xy_grid(path: Path, xy_grid: np.ndarray, x_label: str, y_label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"row col {x_label} {y_label}\n")
        for row_idx in range(xy_grid.shape[0]):
            for col_idx in range(xy_grid.shape[1]):
                x_val, y_val = xy_grid[row_idx, col_idx]
                handle.write(f"{row_idx} {col_idx} {x_val:.10f} {y_val:.10f}\n")


def save_zernike_coefficients(path: Path, coeffs: np.ndarray, mode_labels: list[str], header_prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"mode {header_prefix}_coeff\n")
        for label, coeff in zip(mode_labels, coeffs):
            handle.write(f"{label} {coeff:.10f}\n")


def save_zernike_comparison(
    path: Path,
    fpga_coeffs: np.ndarray,
    hcipy_coeffs: np.ndarray,
    mode_labels: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison = np.column_stack((hcipy_coeffs, fpga_coeffs, fpga_coeffs - hcipy_coeffs))
    np.savetxt(path, comparison, fmt="%.10f", header="hcipy_coeff fpga_coeff error", comments="")


class FpgaControlService(QtCore.QObject):
    log_message = QtCore.Signal(str)
    server_state_changed = QtCore.Signal(str)
    client_state_changed = QtCore.Signal(str)
    run_completed = QtCore.Signal(object)
    error_reported = QtCore.Signal(str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._stop_event = threading.Event()
        self._server_thread: threading.Thread | None = None
        self._preview_thread: threading.Thread | None = None
        self._server_socket: socket.socket | None = None
        self._state_lock = threading.Lock()

    @QtCore.Slot(str, int)
    def start_server(self, bind_ip: str = DEFAULT_BIND_IP, bind_port: int = DEFAULT_BIND_PORT) -> None:
        with self._state_lock:
            if self._server_thread is not None and self._server_thread.is_alive():
                self.log_message.emit("Server is already listening.")
                return

            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.settimeout(1.0)
            try:
                server.bind((bind_ip, bind_port))
                server.listen(5)
            except OSError as exc:
                server.close()
                message = f"Failed to listen on {bind_ip}:{bind_port}: {exc}"
                self.server_state_changed.emit("Failed to listen")
                self.error_reported.emit(message)
                return

            self._stop_event.clear()
            self._server_socket = server
            self._server_thread = threading.Thread(
                target=self._accept_loop,
                args=(server, bind_ip, bind_port),
                daemon=True,
            )
            self._server_thread.start()
            self.server_state_changed.emit(f"Listening on {bind_ip}:{bind_port}")
            self.log_message.emit(f"Listening on {bind_ip}:{bind_port}")

    @QtCore.Slot()
    def stop_server(self) -> None:
        self._stop_event.set()
        server = self._server_socket
        self._server_socket = None
        if server is not None:
            try:
                server.close()
            except OSError:
                pass
        self.server_state_changed.emit("Stopped")
        self.client_state_changed.emit("No client connected")
        self.log_message.emit("Server stopped.")

    @QtCore.Slot()
    def generate_local_preview(self) -> None:
        if self._preview_thread is not None and self._preview_thread.is_alive():
            self.log_message.emit("Local preview is already running.")
            return

        self._preview_thread = threading.Thread(target=self._run_local_preview, daemon=True)
        self._preview_thread.start()

    def _accept_loop(self, server: socket.socket, bind_ip: str, bind_port: int) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    client_socket, address = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._stop_event.is_set():
                        return
                    raise

                client_label = f"{address[0]}:{address[1]}"
                self.client_state_changed.emit(f"Connected: {client_label}")
                self.log_message.emit(f"Accepted connection from {client_label}")
                self._handle_client(client_socket, client_label)
                self.client_state_changed.emit("No client connected")
        except Exception as exc:
            message = f"Server loop failed on {bind_ip}:{bind_port}: {exc}"
            self.server_state_changed.emit("Server error")
            self.error_reported.emit(message)
        finally:
            self.server_state_changed.emit("Stopped")

    def _run_local_preview(self) -> None:
        self.server_state_changed.emit("Generating local preview")
        self.log_message.emit("Generating local preview with the host-side HCIPy model.")
        try:
            simulation, hcipy_estimation = self._build_simulation_bundle()
            fpga_like = run_fpga_like_estimation(
                simulation["image_aber"],
                num_subapertures_side=NUM_SUBAPERTURES_SIDE,
                subaperture_pixels=SUBAPERTURE_PIXELS,
            )
            result = RunResult(
                source="local-preview",
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                quantized_image=np.asarray(fpga_like["quantized_image"], dtype=np.uint8),
                centroids_xy_grid=np.asarray(fpga_like["centroids_grid"], dtype=np.float64),
                slopes_xy_grid=np.asarray(fpga_like["slopes_grid"], dtype=np.float64),
                fpga_zernikes=np.asarray(fpga_like["estimated_coeffs"], dtype=np.float64),
                hcipy_zernikes=np.asarray(hcipy_estimation["estimated_coeffs"], dtype=np.float64),
                mode_labels=list(simulation["mode_labels"]),
                notes=[
                    f"Local preview simulating {NUM_VALID_SUBAPERTURES} valid subapertures.",
                    "Using host-side bit-accurate centroid and slope arithmetic.",
                ],
                simulation_data=simulation,
                hcipy_estimation_data=hcipy_estimation,
                true_coeffs=np.asarray(simulation["true_coeffs"], dtype=np.float64),
            )
            self.run_completed.emit(result)
            self.log_message.emit("Local preview finished.")
        except Exception:
            self.error_reported.emit(traceback.format_exc())
        finally:
            if self._server_socket is not None:
                host, port = self._server_socket.getsockname()[:2]
                self.server_state_changed.emit(f"Listening on {host}:{port}")
            else:
                self.server_state_changed.emit("Stopped")

    def _build_simulation_bundle(self) -> tuple[dict[str, object], dict[str, np.ndarray]]:
        simulation = generate_shwfs_case(
            num_lenslets=NUM_SUBAPERTURES_SIDE,
            num_zernike=NUM_ZERNIKE,
            demo_image_path=None,
        )
        hcipy_estimation = run_hcipy_estimation(
            image=simulation["image_aber"],
            estimator=simulation["estimator"],
            reference_slopes=simulation["slopes_ref"],
            reconstruction_matrix=simulation["reconstruction_matrix"],
            zernike_modes=simulation["zernike_modes"],
            aperture=simulation["aperture"],
            measured_opd_field=simulation["input_opd_field"],
            shwfs=simulation["shwfs"],
        )
        return simulation, hcipy_estimation

    def _handle_client(self, client_socket: socket.socket, client_label: str) -> None:
        client_socket.settimeout(CLIENT_TIMEOUT_SEC)
        try:
            while not self._stop_event.is_set():
                request = recv_line(client_socket)
                if request is None:
                    self.log_message.emit("Client disconnected.")
                    return

                request_text = request.decode("utf-8", errors="strict").strip()
                self.log_message.emit(f"Received command: {request_text}")

                if request_text != "start":
                    self.log_message.emit(f"Ignoring unrecognized command: {request_text}")
                    continue

                self.server_state_changed.emit("Processing FPGA run")
                simulation, hcipy_estimation = self._build_simulation_bundle()

                quantized_image = quantize_shwfs_image(simulation["image_aber"])
                coeff_payload = quantized_image.tobytes(order="C")
                client_socket.sendall(coeff_payload)
                self.log_message.emit("Sent quantized detector frame to the FPGA client.")

                if not wait_for_exact_line(client_socket, b"coeffs_done"):
                    raise RuntimeError("Missing coeffs_done acknowledgement from the client.")
                if not wait_for_exact_line(client_socket, b"compute_done"):
                    raise RuntimeError("Missing compute_done marker from the client.")

                centroid_bytes = recv_exact(client_socket, NUM_PACKED_XY * CENTROID_SIZE)
                if centroid_bytes is None:
                    raise RuntimeError("Client disconnected while sending centroids.")
                client_socket.sendall(b"centroids_done\n")

                slope_bytes = recv_exact(client_socket, NUM_PACKED_XY * SLOPE_SIZE)
                if slope_bytes is None:
                    raise RuntimeError("Client disconnected while sending slopes.")
                client_socket.sendall(b"slopes_done\n")

                zernike_bytes = recv_exact(client_socket, NUM_ZERNIKE * ZERNIKE_SIZE)
                if zernike_bytes is None:
                    raise RuntimeError("Client disconnected while sending zernike coefficients.")
                client_socket.sendall(b"zernike_done\n")

                centroids_q4_23 = decode_q4_23_from_bytes(centroid_bytes, NUM_PACKED_XY)
                slopes_q4_23 = decode_q4_23_from_bytes(slope_bytes, NUM_PACKED_XY)
                zernikes_q4_23 = decode_q4_23_from_bytes(zernike_bytes, NUM_ZERNIKE)

                centroids_xy = unpack_xy_vectors(centroids_q4_23, NUM_VALID_SUBAPERTURES)
                slopes_xy = unpack_xy_vectors(slopes_q4_23, NUM_VALID_SUBAPERTURES)

                # Expansion of valid subapertures back to 16x16 grid for display
                from shwfs_utils import expand_valid_subaperture_array
                valid_indices = simulation["valid_subaperture_indices"]
                
                centroids_expanded = expand_valid_subaperture_array(centroids_xy, valid_indices, NUM_TOTAL_SUBAPERTURES)
                slopes_expanded = expand_valid_subaperture_array(slopes_xy, valid_indices, NUM_TOTAL_SUBAPERTURES)

                centroids_xy_grid = reshape_subaperture_xy(centroids_expanded, NUM_SUBAPERTURES_SIDE)
                slopes_xy_grid = reshape_subaperture_xy(slopes_expanded, NUM_SUBAPERTURES_SIDE)

                result = RunResult(
                    source="fpga",
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    quantized_image=np.asarray(quantized_image, dtype=np.uint8),
                    centroids_xy_grid=np.asarray(centroids_xy_grid, dtype=np.float64),
                    slopes_xy_grid=np.asarray(slopes_xy_grid, dtype=np.float64),
                    fpga_zernikes=np.asarray(zernikes_q4_23, dtype=np.float64),
                    hcipy_zernikes=np.asarray(hcipy_estimation["estimated_coeffs"], dtype=np.float64),
                    mode_labels=list(simulation["mode_labels"]),
                    notes=[
                        "The current wire protocol is client-initiated; the HPS side sends start.",
                        "Coefficient accuracy is compared against the HCIPy estimator for the same synthetic case.",
                    ],
                    client_address=client_label,
                    simulation_data=simulation,
                    hcipy_estimation_data=hcipy_estimation,
                    true_coeffs=np.asarray(simulation["true_coeffs"], dtype=np.float64),
                )
                self._write_result_artifacts(result)
                self.run_completed.emit(result)
                self.log_message.emit("FPGA run finished and result artifacts were updated.")
                self.server_state_changed.emit("Listening for the next FPGA run")
        except Exception:
            self.error_reported.emit(traceback.format_exc())
            self.server_state_changed.emit("Listening for the next FPGA run")
        finally:
            try:
                client_socket.close()
            except OSError:
                pass

    def _write_result_artifacts(self, result: RunResult) -> None:
        RESULT_DUMP_DIR.mkdir(parents=True, exist_ok=True)

        if result.fpga_zernikes is not None:
            np.savetxt(
                RESULT_DUMP_DIR / "fpga_zernikes_float.txt",
                result.fpga_zernikes,
                fmt="%.10f",
                header="zernike_coeff",
                comments="",
            )
            save_zernike_coefficients(
                RESULT_DUMP_DIR / "fpga_zernikes_named.txt",
                result.fpga_zernikes,
                result.mode_labels,
                "fpga",
            )

        if result.hcipy_zernikes is not None:
            save_zernike_coefficients(
                RESULT_DUMP_DIR / "hcipy_zernikes_named.txt",
                result.hcipy_zernikes,
                result.mode_labels,
                "hcipy",
            )

        if result.centroids_xy_grid is not None:
            save_xy_grid(
                RESULT_DUMP_DIR / "fpga_centroids_grid.txt",
                result.centroids_xy_grid,
                "x_centroid",
                "y_centroid",
            )

        if result.slopes_xy_grid is not None:
            save_xy_grid(
                RESULT_DUMP_DIR / "fpga_slopes_grid.txt",
                result.slopes_xy_grid,
                "x_slope",
                "y_slope",
            )

        if result.has_accuracy_comparison:
            save_zernike_comparison(
                RESULT_DUMP_DIR / "zernike_comparison.txt",
                np.asarray(result.fpga_zernikes, dtype=np.float64),
                np.asarray(result.hcipy_zernikes, dtype=np.float64),
                result.mode_labels,
            )