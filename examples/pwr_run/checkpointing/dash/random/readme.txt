compared to unaware
1. The scheduler is aware of existing heterogeneity
2. Jobs get to start on V100s and K80s depending on availability
3. When V100s become idle, randomly promote between new jobs and K80 jobs

scheduler design:
When starting jobs on idle GPUs, the job is not actually started yet
The GPU is allocated to it temporarily
Then a promotion decision has to be made, the new job starting on V100 may not get the allocated GPU with random chance
We can keep doing this for 2-gpu jobs. However, the promotion function needs to adapt 
