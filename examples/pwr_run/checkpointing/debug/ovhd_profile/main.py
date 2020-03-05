import pdb
import time
import os
import subprocess
import re
import random
import json
import numpy as np
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import socket
import argparse
import threading
import _thread
import signal
from datetime import datetime

parser = argparse.ArgumentParser(description='TCP client')
args = parser.parse_args()

queue = []
for i in range(50):
    queue.append(i+1)

queue0 = queue[0:12] # 0 - 11
queue1 = queue[12:24] # 12 - 23
queue2 = queue[24:36] # 24 - 35
queue3 = queue[36:] # 36 - 49

save_time = {}
for item in queue:
    save_time[str(item)] = 0
load_time = {}
for item in queue:
    load_time[str(item)] = 0
total_time = {}
for item in queue:
    total_time[str(item)] = 0
finish_dict = {}
for item in queue:
    finish_dict[str(item)] = 0

K80_node = 'c2180'
V100_node = 'd1020'
host_node = 'c0218'
### also, change .h5 file folder in jobs ###

def send_signal(node, cmd):
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = 10000 if node == K80_node else 10001 
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
                print('waiting for success signal')
                time.sleep(1)
    finally:
        #print('closing socket')
        sock.close()

# resume job
def resume_job(node, gpu, job): # resume_job('c2176', '3', '50')
    cmd = 'resume ' + job + ' gpu ' + gpu
    send_signal(node, cmd)

# start job
def start_job(node, gpu, job):
    cmd = 'start ' + job + ' gpu ' + gpu
    send_signal(node, cmd)   

#################### background thread running TCP socket ########################

def thread_function():
    # here listen on the socket 
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (host_node, 10002)
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
                    global save_time
                    global load_time
                    global total_time

                    if 'param' in data_str:
                        pass
                    elif 'save' in data_str: # 'job50 save 35'
                        job_name = data_str.split(' ')[0]
                        job = job_name.replace('job','')
                        time = data_str.split(' ')[2]
                        save_time[job] = int(time) + 1 # in case this isn't changed
                    elif 'load' in data_str: # 'job50 load 35'
                        job_name = data_str.split(' ')[0]
                        job = job_name.replace('job','')
                        time = data_str.split(' ')[2]
                        load_time[job] = int(time) + 1
                        total_time[job] = save_time[job] + load_time[job]
                    elif 'finish' in data_str: # 'job50 finish'
                        global finish_dict
                        job_name = data_str.split(' ')[0]
                        job = job_name.replace('job','')
                        finish_dict[job] = 1

                    print('received ' + data_str)
                    connection.sendall(b'success')
                    #time.sleep(5)
                else:
                    break
        finally:
            connection.close()

x = threading.Thread(target=thread_function, daemon=True)
x.start()

###############################################################################

######################################################################

def thread0():
    for job in queue0:
        job = str(job)
        print('profile job ' + job + ' on gpu 0')
        # first collect save time
        start_job(K80_node, '0', job)
        while True:
            if save_time[job] != 0:
                break
            else:
                time.sleep(5)
        time.sleep(5)
        print('collected save time ' + str(save_time[job]))
        # then collect load time
        resume_job(V100_node, '0', job)
        while True:
            if finish_dict[job] == 1:
                break
            else:
                time.sleep(5)
        time.sleep(5)
        print('collected load time ' + str(load_time[job]))
x0 = threading.Thread(target=thread0)
x0.start()

def thread1():
   for job in queue1:
       job = str(job)
       print('profile job ' + job + ' on gpu 1')
       # first collect save time
       start_job(K80_node, '1', job)
       while True:
           if save_time[job] != 0:
               break
           else:
               time.sleep(5)
       time.sleep(5)
       print('collected save time ' + str(save_time[job]))
       # then collect load time
       resume_job(V100_node, '1', job)
       while True:
           if finish_dict[job] == 1:
               break
           else:
               time.sleep(5)
       time.sleep(5)
       print('collected load time ' + str(load_time[job]))
x1 = threading.Thread(target=thread1)
x1.start()

def thread2():
   for job in queue2:
       job = str(job)
       print('profile job ' + job + ' on gpu 2')
       # first collect save time
       start_job(K80_node, '2', job)
       while True:
           if save_time[job] != 0:
               break
           else:
               time.sleep(5)
       time.sleep(5)
       print('collected save time ' + str(save_time[job]))
       # then collect load time
       resume_job(V100_node, '2', job)
       while True:
           if finish_dict[job] == 1:
               break
           else:
               time.sleep(5)
       time.sleep(5)
       print('collected load time ' + str(load_time[job]))
x2 = threading.Thread(target=thread2)
x2.start()

def thread3():
   for job in queue3:
       job = str(job)
       print('profile job ' + job + ' on gpu 3')
       # first collect save time
       start_job(K80_node, '3', job)
       while True:
           if save_time[job] != 0:
               break
           else:
               time.sleep(5)
       time.sleep(5)
       print('collected save time ' + str(save_time[job]))
       # then collect load time
       resume_job(V100_node, '3', job)
       while True:
           if finish_dict[job] == 1:
               break
           else:
               time.sleep(5)
       time.sleep(5)
       print('collected load time ' + str(load_time[job]))
x3 = threading.Thread(target=thread3)
x3.start()

x0.join()
x1.join()
x2.join()
x3.join()

print('all threads finished')

# after everything is finished
while True:
    if 0 not in list(finish_dict.values()):
        print('finished all runs')
        break
    else:
        print('error, not all jobs finished')
        time.sleep(10)

with open('save_time.json', 'w') as fp1:
    json.dump(save_time, fp1, sort_keys=True, indent=4)
with open('load_time.json', 'w') as fp1:
    json.dump(load_time, fp1, sort_keys=True, indent=4)
with open('total_time.json', 'w') as fp1:
    json.dump(total_time, fp1, sort_keys=True, indent=4)

