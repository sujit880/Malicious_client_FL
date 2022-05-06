import torch
import requests
import json
from os import getpid
import csv
import glob
from pathlib import Path
import re

debug = False
# Fetch Latest Model Params (StateDict)

def clear_model(url: str, Id: str) -> bool:
    # Send GET request
    r = requests.get(url=url + 'clear/'+Id)
    print("status", r.status_code) 
    if r.status_code != 200:
        print("Server Error: Could not fetch Lock Status.\nTrying to set global params...")
        return False   # implies cant read the data
    # Extract data in json format
    data = r.json()
    
    print("Lock data:->", data)


def fetch_params(url: str, Id: str):
    body = {'data':{
        'id' : Id,
        'pid': getpid()
    }
    }
    # Send GET request
    r = requests.get(url=url+'get/'+Id, json=body)
    print("status", r.status_code) 

    if r.status_code == 404:
        print("Server Error: Could not fetch model.\nTrying to set global params...")
        print("ReplyFetch", r)
        return {}, None, None, False

    print("Reply", r)
    # Extract data in json format
    data = r.json()

    # Check for Iteration Number (-1 Means, No model params is present on Server)
    # if data['Iteration'] == -1:
    #     return {}, data['NPush'], data['ModelUpdateCount'], False
    # else:
    #     if debug:
    print("Global Iteration", data['Iteration'])
    global_params = json.loads(data['ModelParams'])
    print("\nglobal paramas type: ", type(global_params))
    return global_params, data['NPush'], data['ModelID'], True
# remove send gradient method as we are not dealing with gradients in FL

# Send Trained Model Params (StateDict)

# Get Model Lock
def get_model_lock(url: str, Id: str) -> bool:
    # Send GET request
    r = requests.get(url=url + 'getLock/'+Id)
    print("status", r.status_code) 
    if r.status_code != 200:
        print("Server Error: Could not fetch Lock Status.\nTrying to set global params...")
        return False   # implies cant read the data
    # Extract data in json format
    data = r.json()
    
    print("Lock data:->", data)

    return data['LockStatus']

def send_local_update(url: str, params: dict, epochs: int, Id: str):
    body =  {'data':{
        'id' : Id,
        'model': json.dumps(params),
        'pid': getpid(),
        'epochs': epochs
    }
    
    }

    # Send POST request
    r = requests.post(url=url+ 'collect', json=body)

    # Extract data in json format
    data = r.json()
    return data


def send_model_params(url: str, params: dict, lr: float, Id: str):
    body = { 'data':{
        'id' : Id,
        'model': json.dumps(params),
        'learning_rate': lr,
        'pid': getpid()
    }
    }

    # Send POST request
    r = requests.post(url=url+'set', json=body)

    # Extract data in json format
    data = r.json()

    print("data-->", data['Iteration'])
    return json.loads(data['ModelParams']), data['NPush'], data['ModelID'], data['Iteration']
# Convert State Dict List to Tensor
def convert_list_to_tensor(params: dict) -> dict:
    params_ = {}
    for key in params.keys():
        params_[key] = torch.tensor(params[key], dtype=torch.float32)

    return params_


# Convert State Dict Tensors to List
def convert_tensor_to_list(params: dict) -> dict:
    params_ = {}
    for key in params.keys():
        params_[key] = params[key].tolist()

    return params_

def csv_writer(path, data):
    f = open(path, 'a')

    # create the csv writer
    writer = csv.writer(f)

    # write a row to the csv file
    for x in data:
        writer.writerow(x)

    # close the file
    f.close()


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path