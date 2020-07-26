compared to unaware
1. The scheduler is aware of existing heterogeneity
2. Jobs get to start on V100s and K80s depending on availability
3. When V100s become idle, randomly promote between new jobs and K80 jobs

scheduler design:
in 3-level design, the P100 is considered as V100. Whenever P100 is free, job gets promoted to P100 from K80
