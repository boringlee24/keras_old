    log_path = '/scratch/li.baol/tsrbrd_log/job_runs/' + testcase + '/'
    threshold = 0.001
    global pc_job
    global PJCT
    for job in job_list:
        # only check for job outside of practical complete job list
        if job not in pc_job and job != 'idle':
            log_dir = log_path + 'job' + job + '/*'
            dirs = glob.glob(log_dir)
            dirs.sort()
            loss_combine = []
            for tc in dirs:
                
                iterator = EventAccumulator(tc).Reload()
                if len(iterator.Tags()['scalars']) > 0:
                    tag = 'loss' #iterator.Tags()['scalars'][2] # this is tag for loss
                    loss = [item.value for item in iterator.Scalars(tag)]
                    loss_combine += loss
