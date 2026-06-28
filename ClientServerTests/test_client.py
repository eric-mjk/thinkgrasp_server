import socket

# put the server_ip - run the command below
"""
hostname -I


  You may see something like:

  192.168.0.11 172.17.0.1

  Use the LAN IP, usually the one starting with 192.168.x.x, 10.x.x.x, or your lab network IP like 147.46.x.x. Do not
  use 127.0.0.1, because that only means “this same machine.”

"""

SERVER_IP = "192.168.0.11"
PORT = 5050

with socket.create_connection((SERVER_IP, PORT), timeout=5.0) as s:
    print(f"Connected to {SERVER_IP}:{PORT}")
    msg = s.recv(64)
    print(f"Received: {msg.decode()}")
