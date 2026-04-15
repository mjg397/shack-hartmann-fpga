#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Author: David Manouchehri <manouchehri@protonmail.com>
# This script will always echo back data on the UDP port of your choice.
# Useful if you want nmap to report a UDP port as "open" instead of "open|filtered" on a standard scan.
# Works with both Python 2 & 3.

import socket
import os
import time

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

server_address = '127.0.0.1'
server_port = 31337

CHUNK_SIZE = 1024
 
server = (server_address, server_port)
sock.bind(server)
print("Listening on " + server_address + ":" + str(server_port))

with open("shwfs_results.png", "rb") as img:
	content = img.read()

while True:
	payload, client_address = sock.recvfrom(1024)
	print("Received request from: " + str(client_address))

	for i in range(0, len(content), CHUNK_SIZE):
		sent = sock.sendto(bytes("DATA: ", "utf-8") + content[i:i + CHUNK_SIZE], client_address)
		time.sleep(0.0005)

	for _ in range(5):
		sent = sock.sendto(bytes("EOF", "utf-8"), client_address)
		time.sleep(0.01)
	print("Sent whole packet")