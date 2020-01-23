import socket
import sys

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect the socket to the port where the server is listening
server_address = ('d1004', 10000)
print('connecting to {} port {}'.format(*server_address))
sock.connect(server_address)

try:

    # Send data
    message = b'nvidia-smi 0'
    print('sending {!r}'.format(message))
    sock.sendall(message)

    data = sock.recv(16)
    if 'success' in data.decode('utf-8'):
        print('received {!r}'.format(data))
    else:
        raise ValueError('TCP connection is invalid')

finally:
    print('closing socket')
    sock.close()
