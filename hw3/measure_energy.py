import time
import numpy as np
import subprocess
import sys
import os
import fcntl
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, nvmlShutdown

(r, w) = os.pipe()
if os.fork() == 0: # child
    w = open(w, 'w')
    cmd = " ".join(sys.argv[1:]) 

    subprocess.run(cmd, shell=True)
    w.write("yum")
else: # parent
    r = open(r, 'r')
    fcntl.fcntl(r, fcntl.F_SETFL, os.O_NONBLOCK) # otherwise file read() blocks
    num_secs, power_sum = 0, 0
    minute_averaged_powers = []
    nvmlInit()
    gpu = nvmlDeviceGetHandleByIndex(0) #Gaurav's pc only has one recognized devucem so the zeroth device is the 1660 ti
    while (r.read(1) == ""):
        time.sleep(1)
        num_secs += 1
        power_sum += nvmlDeviceGetPowerUsage()
        if num_secs % 60 == 0:
            minute_averaged_powers.append(power_sum / 60)
            power_sum = 0

    print(num_secs)
    print(np.mean(minute_averaged_powers))
    nvmlShutdown()
python measure_energy.py python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander_doubleq.yaml --seed 1
