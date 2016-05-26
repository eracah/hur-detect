#!/usr/bin/env python
__author__ = 'racah'
import os
import sys
def make_run_dir(results_dir):
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
        num = 0
    else:
        if len(os.listdir(results_dir)) > 0:
            run_nums = map(lambda fname: int(fname.split('_')[-1]), os.listdir(results_dir))
            num = max(run_nums) + 1
        else:
            num = 0

    run_dir = os.path.join(results_dir, 'run_' + str(num))
    prev_run_dir = os.path.join(results_dir, 'run_' + str(num -1))
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)
    return run_dir, prev_run_dir


if __name__ == "__main__":
    results_dir = sys.argv[1]
    run_dir, prev_run_dir = make_run_dir(results_dir)
    print run_dir