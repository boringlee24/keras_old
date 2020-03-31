##with open('../unaware/logs/unaware_queue_delay.json', 'r') as fp:
##    baseline_only = json.load(fp)
##with open('../random/logs/random_queue_delay.json', 'r') as fp:
##    random_only = json.load(fp)
##with open('../feedback_inverse/logs/feedback_inverse_queue_delay.json', 'r') as fp:
##    feedback_inverse_only = json.load(fp)
##with open('../final4_new/logs_0.05thres/final4_new_queue_delay.json', 'r') as fp:
##    scheme_only = json.load(fp)
##
##unaware = []
##random = []
##feedback_inverse = []
##scheme = []
##for i in range(len(unaware_only)-1):
##    job = str(i+1)
##    unaware.append(unaware_only[job])
##    random.append(random_only[job])
##    feedback_inverse.append(feedback_inverse_only[job])
##    scheme.append(scheme_only[job])
##
##unaware = np.asarray(unaware)
##random = np.asarray(random)
##feedback_inverse = np.asarray(feedback_inverse)
##scheme = np.asarray(scheme)
##
##cols = zip(unaware)
##with open('queue_delay/unaware_queue_delay.csv', 'w') as f:
##    writer = csv.writer(f)
##    for col in cols:
##        writer.writerow(col)
##cols = zip(random)
##with open('queue_delay/random_queue_delay.csv', 'w') as f:
##    writer = csv.writer(f)
##    for col in cols:
##        writer.writerow(col)
##cols = zip(feedback_inverse)
##with open('queue_delay/feedback_inverse_queue_delay.csv', 'w') as f:
##    writer = csv.writer(f)
##    for col in cols:
##        writer.writerow(col)
##cols = zip(scheme)
##with open('queue_delay/scheme_queue_delay.csv', 'w') as f:
##    writer = csv.writer(f)
##    for col in cols:
##        writer.writerow(col)

