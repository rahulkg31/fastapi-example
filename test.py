import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_response():
    try:
        test_json = {"samples": ["i am rahul. how are you?"]}
        response = requests.post("http://127.0.0.1:5001/predict", json=test_json)
        res = response.text
        return res
    except Exception as e:
        print("Error in thread", e)


if __name__ == '__main__':

    print("Sending parallel requests to load test flask server.....")

    start = time.time()
    processes = []
    with ThreadPoolExecutor() as executor:
        for i in range(100):
            processes.append(executor.submit(get_response))

    for task in as_completed(processes):
        print(task.result())

    print("total time (sec): ", time.time() - start)
