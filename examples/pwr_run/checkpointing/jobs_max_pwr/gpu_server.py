import socket
import sys
import pdb
import subprocess

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the port
server_address = ('d1004', 10000)
print('starting up on {} port {}'.format(*server_address))
sock.bind(server_address)

# Listen for incoming connections
sock.listen(1)

while True:
    # Wait for a connection
    print('waiting for a connection')
    connection, client_address = sock.accept()
    try:
        # Receive the data in small chunks and retransmit it
        while True:
            data = connection.recv(16)
            print('received {!r}'.format(data))
            if data:
                data_str = data.decode('utf-8')
                if 'nvidia-smi' in data_str:
                    if 'nvidia-smi 0' in data_str:
                        cmd = './run.sh job24'
                        subprocess.Popen([cmd], shell=True)                                                                
                connection.sendall(b'success')
            else:
                break

    finally:
        # Clean up the connection
        connection.close()
