import os
import torch
import numpy as np
import warnings


def get_freer_device(verbose=True):
    if torch.cuda.is_available():
        i_gpu = get_freer_gpu(verbose=verbose)
        return torch.device(f"cuda:{i_gpu}")
    else:
        if verbose:
            print("No GPU is available.")
        return torch.device("cpu")


def get_freer_gpu(verbose=True):
    randiden=str(np.random.randint(low=0,high=10000))#random identifier for tmp file 
    os.system(f"nvidia-smi -q -d Memory |grep -A4 GPU|grep Total >tmp_total_{randiden}")
    os.system(f"nvidia-smi -q -d Memory |grep -A4 GPU|grep Reserved >tmp_reserved_{randiden}")
    os.system(f"nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp_used_{randiden}")
    memory_total = [int(x.split()[2]) for x in open(f"tmp_total_{randiden}", "r").readlines()]
    memory_reserved = [int(x.split()[2]) for x in open(f"tmp_reserved_{randiden}", "r").readlines()]
    memory_used = [int(x.split()[2]) for x in open(f"tmp_used_{randiden}", "r").readlines()]
    memory_available = [
        x - y - z for x, y, z in zip(memory_total, memory_reserved, memory_used)
    ]
    if len(memory_available) == 0:
        warnings.warn("No GPU is available.")
        return None
    i_gpu = np.argmax(memory_available)
    if verbose:
        print("memory_available", memory_available)
        print("best GPU:", i_gpu)
    #remove tmp files
    os.system(f"rm tmp_total_{randiden} tmp_reserved_{randiden} tmp_used_{randiden}")
    return i_gpu
