import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor
from timeit import default_timer
from time import sleep
from config import *

START_TIME = default_timer()


def fetch(session, index, cmd_cfg):
    base_url = get_config(cmd_cfg)
    print("fetch().base_url = " + str(base_url) + ", cfg = " + str(cmd_cfg))
    sleep(index * n_user_rate_s)
    START_TIME = default_timer()
    with session.get(base_url) as response:
        data = response.text
        if response.status_code != 200:
            print("FAILURE::{0}".format(base_url))

        elapsed = default_timer() - START_TIME
        time_completed_at = "{:5.6f}s".format(elapsed)
        print("{0:<30} {1:>20}".format(index, time_completed_at))

        return data


async def get_data_asynchronous(cmd_cfg):
    print("{0:<30} {1:>20}".format("Client", "Completed at"))
    with ThreadPoolExecutor(max_workers=10) as executor:
        with requests.Session() as session:
            # Set any session parameters here before calling `fetch`
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(
                    executor,
                    fetch,
                    *(session, i, cmd_cfg) # Allows passing multiple arguments
                )
                for i in range(n_users)
            ]
            for response in await asyncio.gather(*tasks):
                pass


def test_client(cmd_cfg):
    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(get_data_asynchronous(cmd_cfg))
    loop.run_until_complete(future)


def main():
    cmd_cfg = "db-endpoint" if len(sys.argv)<2 else sys.argv[1]
    test_client(cmd_cfg)


if __name__ == "__main__":
    main() 
