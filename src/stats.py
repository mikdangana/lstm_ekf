import sys
from config import *


def stats(cfg_key):
    script = sys.argv[0].replace("stats.py", "client.py")
    os_run("python " + script + " '" + cfg_key + "' > out.txt")
    content = open("out.txt", "r").read()
    print("content = " + str(len(content)))
    lines = content.split("\n") #os_run("python client.py '" + cfg_key + "'").split("\n")
    rows = list(map(lambda l: l.split(), lines))
    rows = filter(lambda r: r[1] if len(r)>1 and "s" in r[1] else None, rows)
    times = list(map(lambda r: float(r[1].replace("s", "")), rows))
    avg = sum(times) / len(times)
    print("stats(" + cfg_key + "): # = " + str(len(times)) + ", average = " + str(avg) + ", min = " + str(min(times)) + ", max = " + str(max(times)) + ", failures = " + str(len(lines) - len(times)))


#stats("db-endpoint")

stats("search-endpoint")

