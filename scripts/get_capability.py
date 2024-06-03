import os
os.chdir("/n/home12/cfpark00/ML/ToyCompDiff")

import torch
import numpy as np
import os
import glob
import tqdm

from mltools.networks import networks
from mltools.utils import cuda_tools

import utils
import models

import sys
supfol=sys.argv[1]

device=cuda_tools.get_freer_device()


classifier_ckpt_path="./data/images_1/2x2_final2/classifier_combined.pth"
n_classes=[2,2]
net=networks.CUNet(shape=(3,32,32),out_channels=64,chs=[32,32,32],norm_groups=4)
classifier=models.Classifier(net=net,n_classes=n_classes)
classifier=classifier.to(device)
classifier.load_state_dict(torch.load(classifier_ckpt_path))
classifier=classifier.eval()

over_conditioning_colors=[[0.4,0.4,0.6],[0.35,0.35,0.65],[0.25,0.25,0.75],[0.15,0.15,0.85],[0.05,0.05,0.95]]

def get_step(ckpt):
    return int(ckpt.split("ckpt_step=")[1].split(".pth")[0])

fols=glob.glob(os.path.join(supfol,"*"))

print(fols)

for fol in fols:
    seed=int(fol.split("seed=")[1])
    config=utils.load_config(glob.glob(os.path.join(fol,"*.yaml"))[0])
    ckpt_paths=glob.glob(os.path.join(fol,"ckpts","*.pth"))
    ckpt_paths=sorted(ckpt_paths,key=get_step)

    model=utils.get_model(config)
    model.to(device)
    model=model.eval()

    plot_data=torch.load(os.path.join(fol,f"plot_data.pth"))
    accs_te=np.array(plot_data['classprobs_pred_te'])[...,1].mean(-1)
    data={}
    data["accs_te"]=accs_te

    steps=[]
    promptaccs=[]
    for i,ckpt_path in tqdm.tqdm(enumerate(ckpt_paths),total=len(ckpt_paths)):
        step=get_step(ckpt_path)
        model.load_state_dict(torch.load(ckpt_path))
        accs=[]
        for over_conditioning_color in over_conditioning_colors:
            gens_=model.generate(c=torch.tensor([0.,0.,0.,0., *over_conditioning_color, 0.3, 0.,0.,0.])[None].repeat(64,1).to(device))
            accs_=torch.stack(classifier.classify(gens_,return_probs=True),dim=0)
            accs.append(accs_[:,:,1].detach().cpu().numpy())
        steps.append(step)
        accs=np.stack(accs,axis=0)
        promptaccs.append(accs)
    steps=np.array(steps)
    promptaccs=np.stack(promptaccs,axis=0)
    data["steps"]=steps
    data["promptaccs"]=promptaccs
    data["fol"]=fol

    save_path=os.path.join(fol,f"prompt_accs3.pth")
    torch.save(data,save_path)