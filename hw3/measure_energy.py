from time import sleep, strftime
import numpy as np
import subprocess, sys, os, fcntl, signal, csv
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, nvmlShutdown

poll_keyword, average_keyword, suffix_keyword, cfg = "--poll_rate", "--averaging_n", "--suffix", "-cfg" #parameters for user
argv = sys.argv
print(argv)
if poll_keyword in argv: #POLLING_RATE IS THE NUMBER OF SECONDS BETWEEN POLLS OF POWER CONSUMPTION
    idx = argv.index(poll_keyword)
    polling_rate = float(argv[idx + 1])
    del argv[idx + 1]
    del argv[idx]
else:
    polling_rate = 1 
if average_keyword in argv: #AVERAGING_N IS THE NUMBER OF POLLS TO BIN OVER
    idx = argv.index(average_keyword)
    averaging_n = int(argv[idx + 1])
    del argv[idx + 1]
    del argv[idx]
else:
    averaging_n = 30
if suffix_keyword in argv: #SUFFIX IS A LOG FILE NAME SUFFIX
    idx = argv.index(suffix_keyword)
    suffix = argv[idx + 1]
    del argv[idx + 1]
    del argv[idx]
else:
    suffix = None 
print(argv)

if cfg in argv:
    idx = argv.index(cfg)
    config_file = argv[idx + 1].split(".")[0].split("/")[-1]
else:
    config_file = "underdetermined_command"

log_file =  config_file + "_" + strftime("%d-%m-%Y_%H-%M-%S")
if suffix:
    log_file += "_" + suffix
print(log_file)

(r, w) = os.pipe()
if os.fork() == 0: # child
    w = open(w, 'w')
    cmd = " ".join(argv[1:])# + " --log_file " + log_file
    print(cmd)
    try:
        p = subprocess.run(cmd, shell=True)
    except KeyboardInterrupt:
        pass
    w.write("yum")
else: # parent
    r = open(r, 'r')
    fcntl.fcntl(r, fcntl.F_SETFL, os.O_NONBLOCK) # otherwise file read() blocks
    num_polls, power_sum = 0, 0
    minute_averaged_powers = []
    nvmlInit()
    gpu = nvmlDeviceGetHandleByIndex(0) #Gaurav's pc only has one recognized device so the zeroth device is the 1660 ti
    
    while (r.read(1) == ""):
        sleep(polling_rate)
        num_polls += 1
        power_sum += nvmlDeviceGetPowerUsage(gpu)
        if num_polls % averaging_n == 0:
            minute_averaged_powers.append(power_sum / averaging_n)
            power_sum = 0
    
    print("number of polls", num_polls)
    print("mean power usage in mw: ", np.mean(minute_averaged_powers))
    nvmlShutdown()

def measure_power(line, polling_interval=1, bin_size=60):
    (r, w) = os.pipe()
    if os.fork() == 0: # child
        w = open(w, 'w')
        try:
            exec(line)
        except KeyboardInterrupt:
            pass
        w.write("yum")
    else: # parent
        r = open(r, 'r')
        fcntl.fcntl(r, fcntl.F_SETFL, os.O_NONBLOCK) # otherwise file read() blocks
        num_polls, power_sum = 0, 0
        averaged_powers = []
        nvmlInit()
        gpu = nvmlDeviceGetHandleByIndex(0) #Gaurav's pc only has one recognized device so the zeroth device is the 1660 ti
        
        while (r.read(1) == ""):
            sleep(polling_interval)
            num_polls += 1
            power_sum += nvmlDeviceGetPowerUsage(gpu)
            if num_polls >= bin_size:
                averaged_powers.append(power_sum / num_polls)
                num_polls, power_sum = 0, 0
        
        print("mean power usage in mw: ", np.mean(averaged_powers))
        nvmlShutdown()
    return np.mean(averaged_powers)