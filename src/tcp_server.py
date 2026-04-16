import socket 
import threading 
import os

bind_ip = "10.48.157.28" 
bind_port = 1234

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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
                print("Executing file send. ")
                file_size = os.path.getsize("file_example_PNG_3MB.png")
                client_socket.sendall(file_size.to_bytes(4, "big"))
                with open("file_example_PNG_3MB.png", "rb") as f:
                    while True:
                        chunk = f.read(4096)
                        if not chunk:
                            break
                        client_socket.sendall(chunk)

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


while True: 
    # When a client connects we receive the 
    # client socket into the client variable, and 
    # the remote connection details into the addr variable
    client, addr = server.accept() 
    print(f"[+] Accepted connection from: {addr[0]}:{addr[1]}")
    #spin up our client thread to handle the incoming data 
    client_handler = threading.Thread(target=handle_client, args=(client,))
    client_handler.start() 