import os
os.chdir("/n/home12/cfpark00/ML/ToyCompDiff")

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import glob
import tqdm
import scipy.ndimage as sim
import os

import utils

import importlib
importlib.reload(utils)

from mltools.utils import cuda_tools
device=cuda_tools.get_freer_device()

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("fol",type=str,help="Folder to regenerate")
    parser.add_argument("suffix",type=str,help="Suffix")
    args=parser.parse_args()
    fol=args.fol
    suffix=args.suffix

    config=utils.load_config(glob.glob(os.path.join(fol,"*.yaml"))[0])
    model=utils.get_model(config)

    n_samples_train_gen=np.array(config["n_samples_train_gen"])
    n_samples_test_gen=np.array(config["n_samples_test_gen"])
    config["n_samples_train_gen"]=n_samples_train_gen
    config["n_samples_test_gen"]=n_samples_test_gen
    ### set attributes
    config["data_params"]["size"]["means"]=[0.45,0.35]
    ###
    x_tr,y_tr,l_tr,x_te,y_te,l_te=utils.generate_data(config,forgen=True)
    ckpt_paths=utils.get_ckpt_paths(fol)

    generation_fol=os.path.join(fol,"generations_"+suffix)
    assert not os.path.exists(generation_fol)
    os.makedirs(generation_fol)

    for step,ckpt_path in tqdm.tqdm(ckpt_paths.items(),total=len(ckpt_paths)):
        model.load_state_dict(torch.load(ckpt_path))
        model=model.to(device)
        model.eval()
        gen_tr=model.generate(torch.tensor(y_tr,dtype=torch.float32,device=device)).detach().cpu().numpy()
        gen_te=model.generate(torch.tensor(y_te,dtype=torch.float32,device=device)).detach().cpu().numpy()
        generation={}
        generation['gen_tr']=gen_tr
        generation['gen_te']=gen_te
        generation['y_tr_gen']=y_tr
        generation['y_te_gen']=y_te
        generation['l_tr_gen']=l_tr
        generation['l_te_gen']=l_te

        generation_path=os.path.join(generation_fol,f"gen_{step}.pth")
        torch.save(generation,generation_path)
