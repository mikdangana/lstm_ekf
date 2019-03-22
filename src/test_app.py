from config import *
from time import sleep

verbose = False

def show(s):
    if verbose:
        print(s)


if __name__ == "__main__":
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    (i, nums) = (0, [10])
    while True: 
        nums = nums + nums
        i += 1
        if i % 25 == 0:
            show("sleeping, nums = " + str(len(nums)) + ", i = " + str(i))
            show("top = " + str(os_run("top -b -n 1  | head -10")))
            sleep(1.25)
            show("slept")
            nums = [10]
