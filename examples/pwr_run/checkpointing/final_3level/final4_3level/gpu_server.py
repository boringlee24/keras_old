import socket
import sys
import pdb
import subprocess
import re
import json
import os
import argparse
#import psutil
import time
import datetime

parser = argparse.ArgumentParser(description='TCP server')
parser.add_argument('--node', metavar='GPU_NODE', type=str, help='specific which node')
parser.add_argument('--port', metavar='PORT_NUMBER', type=int, help='select which port for communication')
parser.add_argument('--tc', metavar='TESTCASE', type=str, help='select testcase')
parser.add_argument('--host', metavar='HOST_NODE', type=str, help='scheduler node')
args = parser.parse_args()

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the port
server_address = (args.node, args.port)
print('starting up on {} port {}'.format(*server_address))
sock.bind(server_address)

# Listen for incoming connections
sock.listen(5)

testcase = args.tc
log_dir = '/scratch/li.baol/jobs_logs/' + testcase + '/' # this is dir for training progress

run_pid_dict = {}

while True:
    # Wait for a connection
    connection, client_address = sock.accept()
    try:
        # keep receiving until nothing is received 
        while True:
            data = connection.recv(32)
            if data:
                data_str = data.decode('utf-8')
                print('received ' + data_str)
                if 'measure' in data_str: # 'measure job20 gpu 3'
                    jobid = re.findall(r'\d+', data_str)[0]
                    gpuid = re.findall(r'\d+', data_str)[1]
                    cmd = './run.sh job' + jobid + ' ' + gpuid + ' ' + testcase
                    print('measuring power for job' + jobid + ' at gpu ' + gpuid)
                    meas_pid = subprocess.Popen([cmd], shell=True).pid
                    run_pid_dict['job'+jobid] = meas_pid
                elif 'start' in data_str: # 'start 15 gpu 2'
                    jobid = re.findall(r'\d+', data_str)[0]
                    out_file = log_dir + jobid + '.out'
                    err_file = log_dir + jobid + '.err'
                    gpuid = re.findall(r'\d+', data_str)[1]
                    cmd = 'python job' + jobid + '.py --tc ' + testcase + ' --gpu_num ' + gpuid + ' --node ' + args.host
                    os.makedirs(os.path.dirname(out_file), exist_ok=True)
                    os.makedirs(os.path.dirname(err_file), exist_ok=True)
                    with open(out_file, 'w+') as out, open(err_file, 'w+') as err:
                        subprocess.Popen([cmd], shell=True, stdout=out, stderr=err)
                    print('starting job' + jobid + ' at gpu ' + gpuid)
                elif 'resume' in data_str: # 'resume 15 gpu 2'
                    jobid = re.findall(r'\d+', data_str)[0]
                    out_file = log_dir + jobid + '.out'
                    err_file = log_dir + jobid + '.err'
                    gpuid = re.findall(r'\d+', data_str)[1]
                    cmd = 'python job' + jobid + '.py --tc ' + testcase + ' --gpu_num ' + gpuid + ' --node ' + args.host + ' --resume'
                    os.makedirs(os.path.dirname(out_file), exist_ok=True)
                    os.makedirs(os.path.dirname(err_file), exist_ok=True)
                    with open(out_file, 'w+') as out, open(err_file, 'w+') as err:
                        subprocess.Popen([cmd], shell=True, stdout=out, stderr=err)
                    print('resuming job' + jobid + ' at gpu ' + gpuid)
                elif 'save' in data_str: # 'save 15 pid 10000'
                    jobid = data_str.split(' ')[1]
                    pid = data_str.split(' ')[3]
                    cmd = 'kill -15 ' + pid
                    print('sending checkpointing command to PID ' + pid)
                    subprocess.Popen([cmd], shell=True)
                    print('checkpointing job' + jobid)
                elif 'kill' in data_str: # 'kill 15', kills the run.sh processes
                    jobid = re.findall(r'\d+', data_str)[0]
                    run_pid = run_pid_dict['job'+jobid]
                    cmd = 'pkill -15 -P ' + str(run_pid)
                    print('sending kill command to run.sh PID ' + str(run_pid))
                    subprocess.Popen([cmd], shell=True)

                connection.sendall(b'success')
            else:
                break

    finally:
        # Clean up the connection
        connection.close()
