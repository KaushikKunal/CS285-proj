from time import sleep, strftime
import numpy as np
import subprocess, sys, os, fcntl, argparse, pandas, psutil, signal
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, nvmlShutdown
from cs285.scripts.scripting_utils import make_logger, make_config

USE_EXCLUSIVE_CORES = True
poll_keyword = "--poll" #parameters for user
argv = sys.argv
if poll_keyword in argv: #POLLING_INTERVAL IS THE NUMBER OF SECONDS BETWEEN POLLS OF POWER CONSUMPTION
    idx = argv.index(poll_keyword)
    polling_interval = float(argv[idx + 1])
    del argv[idx + 1]
    del argv[idx]
else:
    polling_interval = 1 

# print(os.getpid())
# while True:
#     sleep(3)
#     print('hah')

#initialization
nvmlInit()
gpu = nvmlDeviceGetHandleByIndex(0) #Gaurav's pc only has one recognized device so the zeroth device is the 1660 ti
num_polls, power_sum = 0, 0


cmd = " ".join(argv[1:])
if (USE_EXCLUSIVE_CORES):
    cmd = "taskset -c 7 " + cmd  # TODO: hardcoded core to 7 rn
p = subprocess.Popen(cmd, shell=True, stdout=None, stderr=None)
parent = psutil.Process(p.pid)
child_pid = parent.children(recursive=True)[0].pid

#p = subprocess.run(cmd, shell=True)
#process = psutil.Process(p.pid)
#sleep(1)

def get_utilization(pid):
    ps_output = subprocess.run(("ps -p {} -o %cpu,%mem".format(pid)), capture_output=True, text=True, shell=True)
    
    if (len(ps_output.stdout.split('\n')) < 3):
        return 0.0, 0.0
    cpu, mem = ps_output.stdout.split('\n')[1].split()
    return float(cpu), float(mem)

# arrays storing utilization data
cpu_arr = []
mem_arr = []

while p.poll() == None:
    sleep(polling_interval)
    num_polls += 1
    power_sum += nvmlDeviceGetPowerUsage(gpu)
    cpu, mem = get_utilization(child_pid)
    cpu_arr.append(cpu)
    mem_arr.append(mem)
    #print(p.virtual_memory())
    #print(process.cpu_percent(polling_interval))
    

print("number of polls", num_polls)
print("mean power usage in mw: ", power_sum / (num_polls + 1e-5))
nvmlShutdown()

if "cs285/scripts/eval_hw3_dqn.py" in sys.argv:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--no_log", "-nlog", action="store_true")
    parser.add_argument("--config_file", "-cfg", type=str, required=True)
    args, unknown = parser.parse_known_args()
    logdir_prefix = f"eval_dqn_"
    config = make_config(args.config_file)
    if not args.no_log:
        logger = make_logger(logdir_prefix, config, csv=True, latent=True)
        logger_utilization = make_logger(logdir_prefix + "utilization_", config, csv=True, latent=False)
        logger.log_scalar(power_sum / (num_polls + 1e-5), "average_power (mw)")
        logger.log_scalar(num_polls, "number of polls")
        logger.log_scalar(polling_interval, "polling_interval")
        logger_utilization.log_scalar(str(cpu_arr), "cpu%")
        logger_utilization.log_scalar(str(mem_arr), "mem%")
        logger.log_scalar(sum(cpu_arr)/(num_polls + 1e-5), "avg_cpu%")
        logger.log_scalar(sum(mem_arr)/(num_polls + 1e-5), "avg_mem%", display=True)

if "cs285/scripts/run_hw3_dqn.py" in sys.argv:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--no_log", "-nlog", action="store_true")
    parser.add_argument("--config_file", "-cfg", type=str, required=True)
    parser.add_argument("--prune_amount", "-pa", type=float, default=0)
    parser.add_argument("--derank_amount", "-lra", type=float, default=0)
    args, unknown = parser.parse_known_args()
    logdir_prefix = f"train_dqn_"
    config = make_config(args.config_file)
    if not args.no_log:
        logger = make_logger(logdir_prefix, config, csv=True, latent=True)
        logger_utilization = make_logger(logdir_prefix + "utilization_", config, csv=True, latent=False)
        logger.log_scalar(power_sum / (num_polls + 1e-5), "average_power (mw)")
        logger.log_scalar(num_polls, "number of polls")
        logger.log_scalar(polling_interval, "polling_interval")
        logger_utilization.log_scalar(str(cpu_arr), "cpu%")
        logger_utilization.log_scalar(str(mem_arr), "mem%")
        logger.log_scalar(sum(cpu_arr)/(num_polls + 1e-5), "avg_cpu%")
        logger.log_scalar(sum(mem_arr)/(num_polls + 1e-5), "avg_mem%", display=True)
        

    



# from time import sleep, strftime
# import numpy as np
# import subprocess, sys, os, fcntl, argparse, pandas
# from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, nvmlShutdown
# from cs285.scripts.scripting_utils import make_logger, make_config

# poll_keyword = "--poll" #parameters for user
# argv = sys.argv
# if poll_keyword in argv: #POLLING_RATE IS THE NUMBER OF SECONDS BETWEEN POLLS OF POWER CONSUMPTION
#     idx = argv.index(poll_keyword)
#     polling_interval = float(argv[idx + 1])
#     del argv[idx + 1]
#     del argv[idx]
# else:
#     polling_interval = 1 



# (r, w) = os.pipe()
# if os.fork() == 0: # child
#     w = open(w, 'w')
#     cmd = " ".join(argv[1:])
#     try:
#         print("greg")
#         #p = subprocess.run(cmd, shell=True)
#         p = subprocess.Popen(cmd, shell=True, stdout=None, stderr=None)
#         print("huhuh " + str(p.pid))
#         w.write(str(p.pid) + "\n")
#         p.wait()
#         print("john")
#     except KeyboardInterrupt:
#         pass
#     w.write("yum")
# else: # parent
#     r = open(r, 'r')

#     num_polls, power_sum = 0, 0
#     minute_averaged_powers = []
#     nvmlInit()
#     gpu = nvmlDeviceGetHandleByIndex(0) #Gaurav's pc only has one recognized device so the zeroth device is the 1660 ti

#     pid = r.readline() #MUST HAPPEN BEFORE 'fcntl.fcntl' LINE
#     fcntl.fcntl(r, fcntl.F_SETFL, os.O_NONBLOCK) # otherwise file read() blocks
    

#     print("hahah " + str(pid))
#     while (r.read(1) == ""):
#         sleep(polling_interval)
#         num_polls += 1
#         power_sum += nvmlDeviceGetPowerUsage(gpu)
    
#     print("number of polls", num_polls)
#     print("mean power usage in mw: ", power_sum / (num_polls + 1e-5))
#     nvmlShutdown()

#     if "cs285/scripts/eval_hw3_dqn.py" in sys.argv:
#         parser = argparse.ArgumentParser(allow_abbrev=False)
#         parser.add_argument("--prune_amount", "-pa", type=float, default=0)
#         parser.add_argument("--derank_amount", "-lra", type=float, default=0)
#         parser.add_argument("--no_log", "-nlog", action="store_true")
#         parser.add_argument("--config_file", "-cfg", type=str, required=True)
#         args, unknown = parser.parse_known_args()
#         logdir_prefix = f"eval_dqn_prune{args.prune_amount}_rank{args.derank_amount}_"
#         config = make_config(args.config_file)
#         if not args.no_log:
#             logger = make_logger(logdir_prefix, config, csv=True, latent=True)
#             logger.log_scalar(power_sum / (num_polls + 1e-5), "average_power (mw)")
#             logger.log_scalar(num_polls, "number of polls")
#             logger.log_scalar(polling_interval, "polling_interval", display=True)

    


