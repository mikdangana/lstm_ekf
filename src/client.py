import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor
from subprocess import Popen, PIPE
from timeit import default_timer
from time import sleep
from config import *

START_TIME = default_timer()


def fetch(sess, index, cmd_cfg, d = None):
    base_url = get_config(cmd_cfg) if not "http://" in cmd_cfg else cmd_cfg
    base_url = base_url.format(index)
    sleep(index * n_user_rate_s)
    START_TIME = default_timer()
    with requests.post(base_url,data=d) if d else sess.get(base_url) as resp:
        data = resp.text
        if not resp.status_code in [200, 201]:
            print("FAILURE::{0}".format(base_url))

        elapsed = default_timer() - START_TIME
        time_elapsed = "{:5.6f}s".format(elapsed)
        print("{0:<30} {1:>20}".format(START_TIME, time_elapsed))

        return resp, elapsed


async def get_data_asynchronous(cmd_cfg, data=None):
    print("{0:<30} {1:>20}".format("Start-time", "Elapsed"))
    with ThreadPoolExecutor(max_workers=n_client_worker) as executor:
        with requests.Session() as session:
            # Set any session parameters here before calling `fetch`
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(
                    executor,
                    fetch,
                    *(session, i, cmd_cfg, data) # Allows passing multiple args
                )
                for i in range(n_users)
            ]
            res = []
            for response in await asyncio.gather(*tasks):
                res.append(response)
            return res


def test_client(cmd_cfg, data=None):
    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(get_data_asynchronous(cmd_cfg, data))
    return loop.run_until_complete(future)


def get_arg(arg, default):
    return sys.argv[sys.argv.index("-n")+1] if "-n" in sys.argv else default


def main():
    global n_users
    n_users = 5
    n = int(get_arg("-n", n_users))
    (k, f) = (int(get_arg("-n", n*0.3)), int(get_arg("-f", n)))
    n_users = n
    cmd_cfg = "db-endpoint" if len(sys.argv)<2 else sys.argv[1]
    base_url = get_config(cmd_cfg)
    print("fetch().base_url = " + str(base_url) + ", cfg = " + str(cmd_cfg))
    bscript = "/home/ec2-user/src/blockchain/blockchain.py"
    #procs = [Popen(["python3.6", bscript, "-p", str(5000+i)]) for i in range(n)]
    (_, base_url) = (sleep(0.5), "http://localhost:5{0:03}")
    ips = [base_url.format(i) for i in range(n)]
    if "--register" in sys.argv:
        test_client(base_url+"/nodes/register", {'nodes':",".join(ips)})
        test_client(base_url+"/nodes/registersubnet", {'nodes':",".join(ips[0:k])})
    (txs, d, mine) = ([], {'sender':ips[0], 'recipient':ips[1], 'amount':2}, [])
    for _ in range(int(get_arg("-i", 3))):
        n_users = f 
        txs = txs + test_client(base_url+"/transactions/new", d)
        n_users = 1
        mine = mine + test_client(base_url+"/mine")
    #([p.kill() for p in procs], exit())
    print([tx[1] for tx in mine])
    print([tx[1] for tx in txs])


if __name__ == "__main__":
    main() 
