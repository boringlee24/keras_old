import socket
import sys
import time
import select
from datetime import datetime
import threading
import pdb
import signal
import _thread

def send_signal():
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    node = 'c2180' 
    port = 10001
    cmd = 'abc'
    
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

# create the socket that takes interrupt from localhost port 10002
##sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
##server_address = ('localhost', 10002)
##print('starting up on {} port {}'.format(*server_address))
##sock.bind(server_address)
##sock.listen(5)
##pdb.set_trace()
##readlist = [sock]
##readable, writable, errored = select.select(readlist, [], [])

received_data = ''

def thread_function():
    # here listen on the socket 
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('localhost', 10002)
    print('starting up on {} port {}'.format(*server_address))
    sock.bind(server_address)
    sock.listen(5)  
    while True:
        # Wait for a connection
        connection, client_address = sock.accept()      
        try:
            while True:
                data = connection.recv(32)
                if data: 
                    data_str = data.decode('utf-8')
                    global received_data
                    received_data = data_str
                    print('received ' + data_str)
                    connection.sendall(b'success')
                    _thread.interrupt_main()
                else:
                    break
        finally:
            connection.close()

x = threading.Thread(target=thread_function, daemon=True)
x.start()

def ack_int(signalNumber, frame):
    print('signal acknowledged!!!')
    print(received_data)
    time.sleep(1)

signal.signal(signal.SIGINT, ack_int)

while True:
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("date and time =", dt_string)
    time.sleep(5)
