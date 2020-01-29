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
args = parser.parse_args()

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the port
server_address = (args.node, args.port)
print('starting up on {} port {}'.format(*server_address))
sock.bind(server_address)

# Listen for incoming connections
sock.listen(1)

testcase = args.tc
log_dir = '/scratch/li.baol/jobs_logs/' + testcase + '/' # this is dir for training progress

# first clear pid of all jobs
pid_dict = {}
checkpoint_dict = {}

#def busy_sleep(seconds):
#    start_time = datetime.datetime.now()
#    seconds_slept = 0
#    while seconds_slept < seconds:
#        time.sleep(seconds - seconds_slept)
#        seconds_slept = (datetime.datetime.now() - start_time).total_seconds()

#def check_pid(pid):        
#    """ Check For the existence of a unix pid. """
#    try:
#        os.kill(pid, 0)
#    except OSError:
#        return False
#    else:
#        return True

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
                    cmd = './run.sh job' + jobid + ' ' + gpuid
                    print('measuring power for job' + jobid + ' at gpu ' + gpuid)
                    subprocess.Popen([cmd], shell=True)
                elif 'start' in data_str: # 'start 15 gpu 2'
                    jobid = re.findall(r'\d+', data_str)[0]
                    out_file = log_dir + jobid + '.out'
                    err_file = log_dir + jobid + '.err'
                    gpuid = re.findall(r'\d+', data_str)[1]
                    cmd = 'python job' + jobid + '.py --tc ' + testcase + ' --gpu_num ' + gpuid
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
                    cmd = 'python job' + jobid + '.py --tc ' + testcase + ' --gpu_num ' + gpuid + ' --resume'
                    os.makedirs(os.path.dirname(out_file), exist_ok=True)
                    os.makedirs(os.path.dirname(err_file), exist_ok=True)
                    with open(out_file, 'w+') as out, open(err_file, 'w+') as err:
                        subprocess.Popen([cmd], shell=True, stdout=out, stderr=err)
                    print('resuming job' + jobid + ' at gpu ' + gpuid)
                elif 'save' in data_str: # 'save 15'
                    jobid = re.findall(r'\d+', data_str)[0]
                    with open('pid.json', 'r') as fp1:
                        pid_dict = json.load(fp1)
                    pid = pid_dict['job'+jobid]
                    cmd = 'kill -15 ' + str(pid)
                    print('sending checkpointing command to PID ' + str(pid))
                    subprocess.Popen([cmd], shell=True)
                    print('checkpointing job' + jobid)
                       
#                        if not check_pid(pid): # if this pid doesn't exist any more
#                            break
#                        elif psutil.Process(pid).status() != psutil.STATUS_ZOMBIE: # if pid exist but is a zombie
#                            break
#                        else: # wait for it to end or become a zombie
#                            busy_sleep(1)

                connection.sendall(b'success')
            else:
                break

    finally:
        # Clean up the connection
        connection.close()
