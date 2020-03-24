compared to final4_new
1. added guards for epoch waste time
2. fixed profiling of epoch waste time, use equivalent epoch waste time instead
3. make sure epoch waste time is not worse than a whole epoch time plus 2*scheduler interval. (makes sure it's not worse
than the timed scheme)

