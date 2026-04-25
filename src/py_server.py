import socket 
import threading 
import os
from shwfs_utils import generate_aberrated_image
# from e_matrix_lib import generate_e_matrix
from ematrixgen2 import gen_E_matrix
import numpy as np

CENTROID_SIZE = 4
SLOPE_SIZE = 4
ZERNIKE_SIZE = 4
DUMP_DIR = "debug_dumps"
ENABLE_DUMPS = False

bind_ip = "127.0.0.1" 
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
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def recv_line(sock):
    data = bytearray()
    while True:
        ch = sock.recv(1)
        if not ch:
            return None
        if ch == b"\n":
            return bytes(data)
        data.extend(ch)


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

#client handling thread
def handle_client(client_socket): 
    #printing what the client sends 

    while True:

        request = recv_line(client_socket)
        if request is None:
            print("[+] Client disconnected")
            break

        request_s = request.decode("utf-8", errors="strict").strip()
        print(f"[+] Recieved: {request_s}") 

        match request_s:
            case "start":
                # NEED TO DECIDE ON E MATRIX ENTRY NOTATION
                print("Shack Hartmann Generation - E Matrix Generation running. ")
                e_matrix = gen_E_matrix()

                # Match C-side buffer element width (uint32_t/int32_t): 4 bytes per entry.
                e_matrix = np.ascontiguousarray(e_matrix, dtype=np.int32)

                e_matrix_to_send = summarize_array_for_c("E_fp", e_matrix)
                print(f"[E-MATRIX] sending {e_matrix_to_send.nbytes} bytes over TCP")

                e_matrix_payload = e_matrix_to_send.tobytes(order="C")
                dump_bytes("py_sent_e_matrix", e_matrix_payload, elem_size=4, words_per_row=336)
                client_socket.sendall(e_matrix_payload) # send e_matrix
                print("     E Matrix done sending.")

                # wait for ack (done receiving)
                while True:
                    ack = recv_line(client_socket)
                    if ack == b"matrix_done":
                        print("     E Matrix received.")
                        break

                print("Shack Hartmann Generation - Image Abberation running. ")
                coeffs = generate_aberrated_image() # 65536 array of ints
                interp_coeffs = np.interp(coeffs, (coeffs.min(), coeffs.max()), (0, 255)).astype(np.uint8)
                coeff_payload = interp_coeffs.tobytes(order="C")
                dump_bytes("py_sent_coeffs", coeff_payload, elem_size=1, words_per_row=256)

                client_socket.sendall(coeff_payload) # send coefficient array
                print("     Coeffs done sending.")

                while True:
                    ack = recv_line(client_socket)
                    print(ack)
                    if ack == b"coeffs_done":
                        print("     Coeffs received")
                        break

                while True:
                    ack = recv_line(client_socket)
                    if ack == b"compute_done":
                        print("Compute Done. Receiving results.")
                        break

                # Now recieve 512 centroid vector, 512 slope vector, and 10 zernike vector
                print("receiving centroids")
                centroid_bytes = recv_exact(client_socket, 512 * CENTROID_SIZE)

                if centroid_bytes is None:
                    print("Client disconnected before delivering centroid vector. ")
                    break

                dump_bytes("py_recv_centroids", centroid_bytes, elem_size=4, words_per_row=64)

                client_socket.sendall(b"centroids_done\n")
                print("     Centroids received and acked.")

                print("receiving slopes")
                slope_bytes = recv_exact(client_socket, 512 * SLOPE_SIZE)
                if slope_bytes is None:
                    print("Client disconnected before delivering slope vector. ")
                    break

                dump_bytes("py_recv_slopes", slope_bytes, elem_size=4, words_per_row=64)

                client_socket.sendall(b"slopes_done\n")
                print("     Slopes received and acked.")

                print("receiving zernike coefficients")
                zernike_bytes = recv_exact(client_socket, 10 * ZERNIKE_SIZE)
                if zernike_bytes is None:
                    print("Client disconnected before delivering zernike vector. ")
                    break

                dump_bytes("py_recv_zernike", zernike_bytes, elem_size=4, words_per_row=10)

                client_socket.sendall(b"zernike_done\n")
                print("     Zernike coefficients received and acked.")


            case "returning":
                print("Executing file receive")
                size_bytes = recv_exact(client_socket, 4)
                if size_bytes is None:
                    print("[!] Client disconnected before size header")
                    break

                remaining = int.from_bytes(size_bytes, "big")
                with open("received.bin", "wb") as f:
                    while remaining > 0:
                        recvd = client_socket.recv(min(4096, remaining))
                        if not recvd:
                            print("[!] Client disconnected during file transfer")
                            break
                        f.write(recvd)
                        remaining -= len(recvd)
                print("response received!")

            case "done":
                print("Exiting!")
                break
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