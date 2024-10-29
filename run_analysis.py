import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import tqdm
import argparse


import mltools.utils.cuda_tools as cuda_tools
from mltools.networks import networks

device=cuda_tools.get_freer_device()

import utils
import models
import importlib

def get_yaml_path():
    parser=argparse.ArgumentParser()
    parser.add_argument("config_path",type=str,default="")
    args=parser.parse_args()
    yaml_path=args.config_path
    assert os.path.exists(yaml_path)
    assert yaml_path.endswith(".yaml")
    return yaml_path

if __name__ == "__main__":

    regen=False

    suffix=""
    w_cfg=None
    classifier_ckpt_path="./data/images_1/2x2_final2/classifier_combined.pth"
    n_classes=[2,2]

    #suffix=""
    #classifier_ckpt_path="./data/images_1/2x2x2_final/classifier_combined.pth"
    #n_classes=[2,2,2]

    ####
    device=cuda_tools.get_freer_device(verbose=False)
    yaml_path=get_yaml_path()
    print("Running:",yaml_path)
    config=utils.load_config(yaml_path)

    net=networks.CUNet(shape=(3,32,32),out_channels=64,chs=[32,32,32],norm_groups=4)
    classifier=models.Classifier(net=net,n_classes=n_classes)
    classifier=classifier.to(device)
    classifier.load_state_dict(torch.load(classifier_ckpt_path))
    classifier=classifier.eval()

    fol=config["experiment_directory"]

    plot_data_path=os.path.join(fol,"plot_data"+suffix+".pth")
    config_file=glob.glob(os.path.join(fol,"*.yaml"))[0]
    config=utils.load_config(config_file)
    logs_file=os.path.join(fol,"logs.pth")
    logs=torch.load(logs_file)

    x_tr,y_tr,l_tr,x_te,y_te,l_te=utils.generate_data(config,forgen=True)
    n_classes=config["data_params"]["n_classes"]
    #losses
    ckpts=[]
    classprobs_pred_tr=[]
    classprobs_pred_te=[]
    gens_tr=[]
    gens_te=[]

    if regen:
        model=utils.get_model(config)
        if w_cfg is not None:
            model.model.w_cfg=w_cfg
        ckpt_paths=logs["ckpt_paths"]
    
        for ckpt_path in tqdm.tqdm(ckpt_paths):
            ckpt=torch.load(ckpt_path)
            for key in list(ckpt.keys()):
                if key[:4]=="net.":
                    del ckpt[key]
            ckpts.append(ckpt)
            model.load_state_dict(ckpt)
            model=model.to(device)
            model.eval()
            x_tr_torch=torch.tensor(x_tr).to(device=device,dtype=torch.float32)
            y_tr_torch=torch.tensor(y_tr).to(device=device,dtype=torch.float32)
            x_te_torch=torch.tensor(x_te).to(device=device,dtype=torch.float32)
            y_te_torch=torch.tensor(y_te).to(device=device,dtype=torch.float32)
            ##loss
            with torch.no_grad():
                loss_tr=model.get_loss(x=x_tr_torch,c=y_tr_torch,reduction="none")[1]#decomposed loss
                loss_te=model.get_loss(x=x_te_torch,c=y_te_torch,reduction="none")[1]#decomposed loss
            with torch.no_grad():
                gen_tr=model.generate(y_tr_torch)

                classprob_pred_tr=classifier.classify(gen_tr,return_probs=True)
                classprob_pred_tr=[el.detach().cpu().numpy() for el in classprob_pred_tr]

                gen_te=model.generate(y_te_torch)
                classprob_pred_te=classifier.classify(gen_te,return_probs=True)
                classprob_pred_te=[el.detach().cpu().numpy() for el in classprob_pred_te]

                gen_tr=gen_tr.detach().cpu().numpy()
                gen_te=gen_te.detach().cpu().numpy()
            classprobs_pred_tr.append(classprob_pred_tr)
            classprobs_pred_te.append(classprob_pred_te)
            gens_tr.append(gen_tr)
            gens_te.append(gen_te)
    else:
        def get_step(path):
            return int(path.split("_")[-1].split(".")[0])
        generation_paths=glob.glob(os.path.join(fol,"generations","*.pth"))
        generation_paths=sorted(generation_paths,key=get_step)

        for generation_path in tqdm.tqdm(generation_paths):
            generation=torch.load(generation_path)
            gen_tr=torch.tensor(generation["gen_tr"]).to(dtype=torch.float32,device=device)
            gen_te=torch.tensor(generation["gen_te"]).to(dtype=torch.float32,device=device)
            with torch.no_grad():
                classprob_pred_tr=classifier.classify(gen_tr,return_probs=True)
                classprob_pred_tr=[el.detach().cpu().numpy() for el in classprob_pred_tr]
                classprob_pred_te=classifier.classify(gen_te,return_probs=True)
                classprob_pred_te=[el.detach().cpu().numpy() for el in classprob_pred_te]
                gen_tr=gen_tr.detach().cpu().numpy()
                gen_te=gen_te.detach().cpu().numpy()
            classprobs_pred_tr.append(classprob_pred_tr)
            classprobs_pred_te.append(classprob_pred_te)
            gens_tr.append(gen_tr)
            gens_te.append(gen_te)
    gens_tr=np.stack(gens_tr,axis=0)
    gens_te=np.stack(gens_te,axis=0)
    if suffix!="":
        gens_path=os.path.join(fol,"gens"+suffix+".pth")
        torch.save({"gens_tr":gens_tr,"gens_te":gens_te},gens_path)

    rightprobss_tr=[]
    rightprobss_te=[]
    rights_tr=[]
    rights_te=[]
    indarr_tr=np.arange(len(l_tr))
    right_classes_tr=classifier.unravel_index(l_tr)
    indarr_te=np.arange(len(l_te))
    right_classes_te=classifier.unravel_index(l_te)
    for i_step in range(len(classprobs_pred_tr)):
        rightprobs_tr=[]
        rightprobs_te=[]
        rights_tr_=[]
        rights_te_=[]
        for i_con in range(len(right_classes_tr)):
            rightprob_tr=classprobs_pred_tr[i_step][i_con][indarr_tr,right_classes_tr[i_con]]
            rightprobs_tr.append(rightprob_tr)
            rights_tr_.append(classprobs_pred_tr[i_step][i_con].argmax(-1)==right_classes_tr[i_con])
            rightprob_te=classprobs_pred_te[i_step][i_con][indarr_te,right_classes_te[i_con]]
            rightprobs_te.append(rightprob_te)
            rights_te_.append(classprobs_pred_te[i_step][i_con].argmax(-1)==right_classes_te[i_con])
        rightprobss_tr.append(np.stack(rightprobs_tr,axis=-1))
        rights_tr.append(np.stack(rights_tr_,axis=-1))
        rightprobss_te.append(np.stack(rightprobs_te,axis=-1))
        rights_te.append(np.stack(rights_te_,axis=-1))
    rightprobss_tr=np.stack(rightprobss_tr,axis=0)
    rights_tr=np.stack(rights_tr,axis=0)
    rightprobss_te=np.stack(rightprobss_te,axis=0)
    rights_te=np.stack(rights_te,axis=0)

    plot_data={}
    plot_data["min_vlb"]=logs.get("min_vlb",0)
    plot_data["save_steps"]=logs["save_steps"]
    plot_data["losses"]=logs["losses"]
    plot_data["val_losses"]=logs["val_losses"]
    plot_data["te_losses"]=logs["te_losses"]
    plot_data["l_tr"]=l_tr
    plot_data["rights_tr"]=rights_tr
    plot_data["l_te"]=l_te
    plot_data["rights_te"]=rights_te
    plot_data["classprobs_pred_tr"]=classprobs_pred_tr
    plot_data["classprobs_pred_te"]=classprobs_pred_te
    plot_data["rightprobss_tr"]=rightprobss_tr
    plot_data["rightprobss_te"]=rightprobss_te
    torch.save(plot_data,plot_data_path)


