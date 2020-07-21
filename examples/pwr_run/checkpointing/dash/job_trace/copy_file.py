from shutil import copyfile

for i in range(4, 51):
    src = 'jobs/job'+str(i+50)+'.py'
    dst = 'jobs_50/job'+str(i)+'.py'
    copyfile(src, dst)

