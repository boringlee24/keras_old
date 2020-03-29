from shutil import copyfile

for i in range(1, 51):
    src = 'job'+str(i)+'.py'
    dst = 'job'+str(i+50)+'.py'
    copyfile(src, dst)

