import math
import time

def simulate_task():
    for _ in range(10**7):
        math.sqrt(_)

if __name__ == "__main__":
    while True:
        simulate_task()
        time.sleep(1)