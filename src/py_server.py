import socket 
import threading 
import os
from shwfs_utils import (
    build_fpga_estimation,
    collapse_xy_grid,
    expand_valid_subaperture_array,
    generate_shwfs_case,
    generate_shwfs_visualizations,
    reshape_subaperture_xy,
    run_hcipy_estimation,
)
# from e_matrix_lib import generate_e_matrix
from ematrixgen2 import gen_E_matrix
import numpy as np

CENTROID_SIZE = 4
SLOPE_SIZE = 4
ZERNIKE_SIZE = 4
DUMP_DIR = "debug_dumps"
ENABLE_DUMPS = False
RESULT_DUMP_DIR = "output_dumps"
CLIENT_TIMEOUT_SEC = 5.0

NUM_SUBAPERTURES_SIDE = 16
NUM_SUBAPERTURES = NUM_SUBAPERTURES_SIDE * NUM_SUBAPERTURES_SIDE
SUBAPERTURE_PIXELS = 16
NUM_PACKED_XY = NUM_SUBAPERTURES * 2
NUM_ZERNIKE = 10
Q4_23_SCALE = float(1 << 23)

bind_ip = "10.48.178.226" 
bind_port = 80

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.settimeout(1.0)
server.bind((bind_ip, bind_port)) 
# we tell the server to start listening with 
# a maximum backlog of connections set to 5
server.listen(5) 

print(f"[+] Listening on port {bind_ip} : {bind_port}")                            

if ENABLE_DUMPS:
    os.makedirs(DUMP_DIR, exist_ok=True)

os.makedirs(RESULT_DUMP_DIR, exist_ok=True)


def summarize_array_for_c(name, arr):
    a = np.ascontiguousarray(arr)
    flat = a.ravel(order="C")

    if a.dtype.byteorder == "<":
        byteorder_s = "little-endian"
    elif a.dtype.byteorder == ">":
        byteorder_s = "big-endian"
    elif a.dtype.byteorder == "=":
        byteorder_s = "native-endian"
    else:
        byteorder_s = "not-applicable"

    c_type_map = {
        np.dtype(np.int8): "int8_t",
        np.dtype(np.uint8): "uint8_t",
        np.dtype(np.int16): "int16_t",
        np.dtype(np.uint16): "uint16_t",
        np.dtype(np.int32): "int32_t",
        np.dtype(np.uint32): "uint32_t",
        np.dtype(np.int64): "int64_t",
        np.dtype(np.uint64): "uint64_t",
        np.dtype(np.float32): "float",
        np.dtype(np.float64): "double",
    }
    c_type = c_type_map.get(a.dtype, "/* unknown */")

    print(f"[E-MATRIX] name: {name}")
    print(f"[E-MATRIX] shape: {a.shape} (ndim={a.ndim})")
    print(f"[E-MATRIX] dtype: {a.dtype}, itemsize: {a.itemsize} bytes, byte-order: {byteorder_s}")
    print(f"[E-MATRIX] elements: {a.size}, payload bytes: {a.nbytes}")
    print(f"[E-MATRIX] C-contiguous: {a.flags['C_CONTIGUOUS']}, send order: row-major (C)")
    print(f"[E-MATRIX] min/max: {a.min()} / {a.max()}")

    preview_count = min(12, flat.size)
    preview_vals = ", ".join(str(v) for v in flat[:preview_count])
    print(f"[E-MATRIX] first {preview_count} values in TCP order: [{preview_vals}]")

    if a.ndim == 2:
        rows, cols = a.shape
        print(f"[E-MATRIX] rows={rows}, cols={cols}")
        print(f"[E-MATRIX] flatten index rule: flat_idx = row * {cols} + col")
        print(f"[E-MATRIX] C hint: {c_type} e_matrix[{rows}][{cols}];")
    else:
        print(f"[E-MATRIX] length={a.shape[0] if a.ndim == 1 else a.size}")
        print(f"[E-MATRIX] C hint: {c_type} e_matrix[{a.size}];")

    return a


def recv_exact(sock, n):
    data = bytearray()
    while len(data) < n:
        try:
            chunk = sock.recv(n - len(data))
        except socket.timeout:
            return None
        except OSError:
            return None
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def recv_line(sock):
    data = bytearray()
    while True:
        try:
            ch = sock.recv(1)
        except socket.timeout:
            return None
        except OSError:
            return None
        if not ch:
            return None
        if ch == b"\n":
            return bytes(data)
        data.extend(ch)


def wait_for_exact_line(sock, expected, max_lines=8):
    for _ in range(max_lines):
        line = recv_line(sock)
        if line is None:
            return False
        if line == expected:
            return True
        print(f"[!] Unexpected line while waiting for {expected!r}: {line!r}")
    print(f"[!] Did not receive {expected!r} within {max_lines} lines")
    return False


def dump_bytes(base_name, payload, elem_size=1, words_per_row=16):
    if not ENABLE_DUMPS:
        return

    bin_path = os.path.join(DUMP_DIR, f"{base_name}.bin")
    txt_path = os.path.join(DUMP_DIR, f"{base_name}.txt")

    with open(bin_path, "wb") as f:
        f.write(payload)

    with open(txt_path, "w") as f:
        if elem_size <= 0 or len(payload) % elem_size != 0:
            # Fallback: raw byte hex dump
            for i, b in enumerate(payload):
                if i > 0 and i % words_per_row == 0:
                    f.write("\n")
                f.write(f"{b:02X} ")
            f.write("\n")
            return

        n_words = len(payload) // elem_size
        hex_digits = elem_size * 2
        for i in range(n_words):
            if i > 0 and i % words_per_row == 0:
                f.write("\n")
            word = int.from_bytes(payload[i * elem_size:(i + 1) * elem_size], "little", signed=False)
            f.write(f"{word:0{hex_digits}X} ")
        f.write("\n")


def decode_q4_23_from_bytes(payload, expected_count):
    values_i32 = np.frombuffer(payload, dtype="<i4", count=expected_count)
    return values_i32.astype(np.float64) / Q4_23_SCALE


def unpack_xy_vectors(packed_values, n_vectors):
    if packed_values.size != (2 * n_vectors):
        raise ValueError(
            f"Expected {2 * n_vectors} packed XY values, got {packed_values.size}"
        )

    x_vals = packed_values[:n_vectors]
    y_vals = packed_values[n_vectors:]
    return np.column_stack((x_vals, y_vals))


def save_xy_grid(path, xy_grid, x_label, y_label):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"row col {x_label} {y_label}\n")
        for row_idx in range(xy_grid.shape[0]):
            for col_idx in range(xy_grid.shape[1]):
                x_val, y_val = xy_grid[row_idx, col_idx]
                handle.write(f"{row_idx} {col_idx} {x_val:.10f} {y_val:.10f}\n")


def save_zernike_coefficients(path, coeffs, mode_labels, header_prefix):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"mode {header_prefix}_coeff\n")
        for label, coeff in zip(mode_labels, coeffs):
            handle.write(f"{label} {coeff:.10f}\n")


def dump_fpga_float_results(slopes_xy_grid, centroids_xy_grid, zernikes, mode_labels):
    np.savetxt(
        os.path.join(RESULT_DUMP_DIR, "fpga_zernikes_float.txt"),
        zernikes,
        fmt="%.10f",
        header="zernike_coeff",
        comments="",
    )
    save_xy_grid(
        os.path.join(RESULT_DUMP_DIR, "fpga_slopes_grid.txt"),
        slopes_xy_grid,
        "x_slope",
        "y_slope",
    )
    save_xy_grid(
        os.path.join(RESULT_DUMP_DIR, "fpga_centroids_grid.txt"),
        centroids_xy_grid,
        "x_centroid",
        "y_centroid",
    )
    save_zernike_coefficients(
        os.path.join(RESULT_DUMP_DIR, "fpga_zernikes_named.txt"),
        zernikes,
        mode_labels,
        "fpga",
    )


def dump_hcipy_reference_results(slopes_xy_grid, centroids_xy_grid, zernikes, mode_labels):
    save_xy_grid(
        os.path.join(RESULT_DUMP_DIR, "hcipy_slopes_grid.txt"),
        slopes_xy_grid,
        "x_slope",
        "y_slope",
    )
    save_xy_grid(
        os.path.join(RESULT_DUMP_DIR, "hcipy_centroids_grid.txt"),
        centroids_xy_grid,
        "x_centroid",
        "y_centroid",
    )
    save_zernike_coefficients(
        os.path.join(RESULT_DUMP_DIR, "hcipy_zernikes_named.txt"),
        zernikes,
        mode_labels,
        "hcipy",
    )


def dump_comparison_results(fpga_slopes_grid, hcipy_slopes_grid, fpga_centroids_grid, hcipy_centroids_grid):
    save_xy_grid(
        os.path.join(RESULT_DUMP_DIR, "slope_error_grid.txt"),
        fpga_slopes_grid - hcipy_slopes_grid,
        "x_error",
        "y_error",
    )
    save_xy_grid(
        os.path.join(RESULT_DUMP_DIR, "centroid_error_grid.txt"),
        fpga_centroids_grid - hcipy_centroids_grid,
        "x_error",
        "y_error",
    )


def summarize_xy_error(name, fpga_xy, hcipy_xy, valid_mask_flat):
    error = fpga_xy - hcipy_xy
    valid_error = error[valid_mask_flat]
    rms_x = np.sqrt(np.mean(valid_error[:, 0] ** 2))
    rms_y = np.sqrt(np.mean(valid_error[:, 1] ** 2))
    rms_mag = np.sqrt(np.mean(np.sum(valid_error ** 2, axis=1)))
    print(
        f"     {name} RMS error over valid subapertures: "
        f"x={rms_x:.6e}, y={rms_y:.6e}, |xy|={rms_mag:.6e}"
    )


def dump_zernike_comparison(fpga_coeffs, hcipy_coeffs, mode_labels):
    comparison = np.column_stack((hcipy_coeffs, fpga_coeffs, fpga_coeffs - hcipy_coeffs))
    header = "hcipy_coeff fpga_coeff error"
    np.savetxt(
        os.path.join(RESULT_DUMP_DIR, "zernike_comparison.txt"),
        comparison,
        fmt="%.10f",
        header=header,
        comments="",
    )
    for label, hcipy_coeff, fpga_coeff in zip(mode_labels, hcipy_coeffs, fpga_coeffs):
        print(
            f"     {label:<12} HCIPy={hcipy_coeff:.6e} "
            f"FPGA={fpga_coeff:.6e} err={(fpga_coeff - hcipy_coeff):.6e}"
        )

#client handling thread
def handle_client(client_socket): 
    #printing what the client sends 
    client_socket.settimeout(CLIENT_TIMEOUT_SEC)

    while True:

        request = recv_line(client_socket)
        if request is None:
            print("[+] Client disconnected")
            break

        request_s = request.decode("utf-8", errors="strict").strip()
        print(f"[+] Recieved: {request_s}") 

        match request_s:
            case "start":

                print("Shack Hartmann Generation - Image Abberation running. ")
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

                coeffs = np.asarray(simulation["image_aber"])
                interp_coeffs = np.interp(coeffs, (coeffs.min(), coeffs.max()), (0, 255)).astype(np.uint8)
                coeff_payload = interp_coeffs.tobytes(order="C")
                dump_bytes("py_sent_coeffs", coeff_payload, elem_size=1, words_per_row=256)

                client_socket.sendall(coeff_payload) # send coefficient array
                print("     Coeffs done sending.")

                if not wait_for_exact_line(client_socket, b"coeffs_done"):
                    print("[!] Missing coeffs_done ack; closing client.")
                    break
                print("     Coeffs received")

                print("Waiting on compute.")

                if not wait_for_exact_line(client_socket, b"compute_done"):
                    print("[!] Missing compute_done marker; closing client.")
                    break
                print("     Compute Done. Receiving results.")

                # Now recieve 512 centroid vector, 512 slope vector, and 10 zernike vector
                print("receiving centroids")
                centroid_bytes = recv_exact(client_socket, NUM_PACKED_XY * CENTROID_SIZE)

                if centroid_bytes is None:
                    print("Client disconnected before delivering centroid vector. ")
                    break

                dump_bytes("py_recv_centroids", centroid_bytes, elem_size=4, words_per_row=64)

                client_socket.sendall(b"centroids_done\n")
                print("     Centroids received and acked.")

                print("receiving slopes")
                slope_bytes = recv_exact(client_socket, NUM_PACKED_XY * SLOPE_SIZE)
                if slope_bytes is None:
                    print("Client disconnected before delivering slope vector. ")
                    break

                dump_bytes("py_recv_slopes", slope_bytes, elem_size=4, words_per_row=64)

                client_socket.sendall(b"slopes_done\n")
                print("     Slopes received and acked.")

                print("receiving zernike coefficients")
                zernike_bytes = recv_exact(client_socket, NUM_ZERNIKE * ZERNIKE_SIZE)
                if zernike_bytes is None:
                    print("Client disconnected before delivering zernike vector. ")
                    break

                dump_bytes("py_recv_zernike", zernike_bytes, elem_size=4, words_per_row=10)

                client_socket.sendall(b"zernike_done\n")
                print("     Zernike coefficients received and acked.")

                # Decode returned Q4.23 fixed-point values into float arrays.
                centroids_q423 = decode_q4_23_from_bytes(centroid_bytes, NUM_PACKED_XY)
                slopes_q423 = decode_q4_23_from_bytes(slope_bytes, NUM_PACKED_XY)
                zernikes_q423 = decode_q4_23_from_bytes(zernike_bytes, NUM_ZERNIKE)

                # Packed format: [X0..X255, Y0..Y255] for both centroids and slopes.
                # The transport order is row-major across the 16x16 lenslet grid, and
                # each lenslet integrates a 16x16 detector-pixel tile.
                centroids_xy = unpack_xy_vectors(centroids_q423, NUM_SUBAPERTURES)
                slopes_xy = unpack_xy_vectors(slopes_q423, NUM_SUBAPERTURES)

                centroids_xy_grid = reshape_subaperture_xy(centroids_xy, NUM_SUBAPERTURES_SIDE)
                slopes_xy_grid = reshape_subaperture_xy(slopes_xy, NUM_SUBAPERTURES_SIDE)

                hcipy_centroids_32 = expand_valid_subaperture_array(
                    hcipy_estimation["slopes_aber"].T,
                    simulation["valid_subaperture_mask"],
                    fill_value=np.nan,
                )
                hcipy_slopes_32 = expand_valid_subaperture_array(
                    hcipy_estimation["slopes_delta"].T,
                    simulation["valid_subaperture_mask"],
                    fill_value=np.nan,
                )

                hcipy_centroids_grid = collapse_xy_grid(
                    hcipy_centroids_32.reshape(NUM_SUBAPERTURES_SIDE * 2, NUM_SUBAPERTURES_SIDE * 2, 2),
                    factor=2,
                )
                hcipy_slopes_grid = collapse_xy_grid(
                    hcipy_slopes_32.reshape(NUM_SUBAPERTURES_SIDE * 2, NUM_SUBAPERTURES_SIDE * 2, 2),
                    factor=2,
                )
                hcipy_centroids_xy = hcipy_centroids_grid.reshape(NUM_SUBAPERTURES, 2)
                hcipy_slopes_xy = hcipy_slopes_grid.reshape(NUM_SUBAPERTURES, 2)
                valid_mask_flat = np.asarray(simulation["fpga_subaperture_mask"], dtype=bool).ravel()

                dump_fpga_float_results(
                    slopes_xy_grid,
                    centroids_xy_grid,
                    zernikes_q423,
                    simulation["mode_labels"],
                )
                dump_hcipy_reference_results(
                    hcipy_slopes_grid,
                    hcipy_centroids_grid,
                    hcipy_estimation["estimated_coeffs"],
                    simulation["mode_labels"],
                )
                dump_comparison_results(
                    slopes_xy_grid,
                    hcipy_slopes_grid,
                    centroids_xy_grid,
                    hcipy_centroids_grid,
                )
                dump_zernike_comparison(
                    zernikes_q423,
                    hcipy_estimation["estimated_coeffs"],
                    simulation["mode_labels"],
                )

                summarize_xy_error("Centroid", centroids_xy, hcipy_centroids_xy, valid_mask_flat)
                summarize_xy_error("Slope", slopes_xy, hcipy_slopes_xy, valid_mask_flat)

                fpga_estimation = build_fpga_estimation(
                    slopes_xy=slopes_xy,
                    estimated_coeffs=zernikes_q423,
                    zernike_modes=simulation["zernike_modes"],
                    aperture=simulation["aperture"],
                    shwfs=simulation["shwfs"],
                    num_lenslets=NUM_SUBAPERTURES_SIDE,
                    measured_opd_field=simulation["input_opd_field"],
                )
                figure_info = generate_shwfs_visualizations(
                    image_ref=simulation["image_ref"],
                    image_aber=simulation["image_aber"],
                    estimation=fpga_estimation,
                    aperture=simulation["aperture"],
                    input_opd_field=simulation["input_opd_field"],
                    true_coeffs=hcipy_estimation["estimated_coeffs"],
                    mode_labels=simulation["mode_labels"],
                    wavelength=simulation["wavelength"],
                    num_lenslets=NUM_SUBAPERTURES_SIDE,
                    results_path=os.path.join(RESULT_DUMP_DIR, "fpga_results.png"),
                    ao_demo_path=os.path.join(RESULT_DUMP_DIR, "fpga_ao_demo.png"),
                    show_plots=False,
                    comparison_label="HCIPy",
                    estimated_label="FPGA",
                    figure_title="Shack-Hartmann WFS Simulation (FPGA vs HCIPy)",
                )

                print(f"     Float outputs written under: {RESULT_DUMP_DIR}")
                print(
                    f"     slope grid shape={slopes_xy_grid.shape}, "
                    f"centroid grid shape={centroids_xy_grid.shape}, "
                    f"zernike shape={zernikes_q423.shape}"
                )
                print(
                    f"     visualization files: {figure_info['results_path']}, "
                    f"{figure_info['ao_demo_path']}"
                )

            case _:
                print("Pattern not recognized")
                # client_socket.close()
        
    client_socket.close()


try:
    while True: 
        try:
            client, addr = server.accept() 
        except socket.timeout:
            continue
        print(f"[+] Accepted connection from: {addr[0]}:{addr[1]}")
        client_handler = threading.Thread(target=handle_client, args=(client,), daemon=True)
        client_handler.start()
except KeyboardInterrupt:
    print("\n[+] Shutting down.")
    server.close() 
