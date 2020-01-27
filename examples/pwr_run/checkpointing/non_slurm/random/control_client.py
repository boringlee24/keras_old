import socket
import sys
import time

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
node = sys.argv[1]
port = sys.argv[2]
cmd = sys.argv[3]

# Connect the socket to the port where the server is listening
server_address = (node, int(port))
print('connecting to {} port {}'.format(*server_address))
sock.connect(server_address)

try:
    # Send data
    message = cmd.encode('utf-8') #b'save 35'  #b'start 35 gpu 6'#b'save 35' 
    print('sending {!r}'.format(message))
    sock.sendall(message)

    while True:
        data = sock.recv(32)
        if 'success' in data.decode('utf-8'):
            print('received {!r}'.format(data))
            break
        else:
            #raise ValueError('TCP connection is invalid')
            print('waiting for success signal')
            time.sleep(1)

finally:
    #print('closing socket')
    sock.close()
