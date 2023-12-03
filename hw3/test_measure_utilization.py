import subprocess

def get_utilization(pid):
    ps_output = subprocess.run(("ps -p {} -o %cpu,%mem".format(pid)), capture_output=True, text=True, shell=True)
    cpu, mem = ps_output.stdout.split('\n')[1].split()
    return cpu, mem

cpu, mem = get_utilization(5550)
print(cpu)
print(mem)