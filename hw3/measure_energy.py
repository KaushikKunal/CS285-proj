from time import sleep, strftime
import numpy as np
import subprocess, sys, os, fcntl, argparse, pandas
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, nvmlShutdown
from cs285.scripts.scripting_utils import make_logger, make_config

poll_keyword = "--poll_interval" #parameters for user
argv = sys.argv
print(argv)
if poll_keyword in argv: #POLLING_RATE IS THE NUMBER OF SECONDS BETWEEN POLLS OF POWER CONSUMPTION
    idx = argv.index(poll_keyword)
    polling_interval = float(argv[idx + 1])
    del argv[idx + 1]
    del argv[idx]
else:
    polling_interval = 1 
print(argv)



(r, w) = os.pipe()
if os.fork() == 0: # child
    w = open(w, 'w')
    cmd = " ".join(argv[1:])
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
        sleep(polling_interval)
        num_polls += 1
        power_sum += nvmlDeviceGetPowerUsage(gpu)
    
    print("number of polls", num_polls)
    print("mean power usage in mw: ", power_sum / num_polls)
    nvmlShutdown()

    if "cs285/scripts/eval_hw3_dqn.py" in sys.argv:
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument("--prune_amount", "-pa", type=float, default=0)
        parser.add_argument("--derank_amount", "-lra", type=float, default=0)
        parser.add_argument("--no_log", "-nlog", action="store_true")
        parser.add_argument("--config_file", "-cfg", type=str, required=True)
        args, unknown = parser.parse_known_args()
        logdir_prefix = f"eval_dqn_prune{args.prune_amount}_rank{args.derank_amount}_"
        config = make_config(args.config_file)
        if not args.no_log:
            logger = make_logger(logdir_prefix, config, csv=True, latent=True)
            logger.log_scalar(power_sum / num_polls, "average_power (mw)")
            logger.log_scalar(num_polls, "number of polls")
            logger.log_scalar(polling_interval, "polling_interval", display=True)


