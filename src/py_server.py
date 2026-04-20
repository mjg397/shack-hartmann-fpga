import socket 
import threading 
import os
from shwfs_utils import generate_aberrated_image
from e_matrix_lib import generate_e_matrix
import numpy as np

CENTROID_SIZE = 4
SLOPE_SIZE = 4
ZERNIKE_SIZE = 4

bind_ip = "192.168.86.45" 
bind_port = 80

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.settimeout(1.0)
server.bind((bind_ip, bind_port)) 
# we tell the server to start listening with 
# a maximum backlog of connections set to 5
server.listen(5) 

print(f"[+] Listening on port {bind_ip} : {bind_port}")                            


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
                e_matrix = generate_e_matrix()
                with open("sent_e_matrix.txt", "w") as f:
                    for i, val in enumerate(e_matrix):
                        if i > 0 and i % 336 == 0:
                            f.write("\n")
                        f.write(f"{val: .8e} ")
                    f.write("\n")
                client_socket.sendall(e_matrix.tobytes()) # send e_matrix
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
                # with open("sent_coeffs.txt", "w") as f:
                #     for i, val in enumerate(interp_coeffs):
                #         if i > 0 and i % 256 == 0:
                #             f.write("\n")
                #         f.write(f"{val:3d} ")
                #     f.write("\n")

                client_socket.sendall(interp_coeffs.tobytes()) # send coefficient array
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