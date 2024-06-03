import yaml
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import shutil
import copy
import tqdm

from mltools.utils import cuda_tools
from mltools import ml_utils

import utils

def get_yaml_path():
    parser=argparse.ArgumentParser()
    parser.add_argument("config_path",type=str,default="")
    args=parser.parse_args()
    yaml_path=args.config_path
    assert os.path.exists(yaml_path)
    assert yaml_path.endswith(".yaml")
    return yaml_path

if __name__ == "__main__":
    device=cuda_tools.get_freer_device(verbose=False)
    yaml_path=get_yaml_path()
    print("Running:",yaml_path)
    config=utils.load_config(yaml_path)
    experiment_directory=config["experiment_directory"]
    if os.path.exists(experiment_directory):
        shutil.rmtree(experiment_directory)
    os.makedirs(experiment_directory)
    shutil.copyfile(yaml_path,os.path.join(experiment_directory,os.path.split(yaml_path)[-1]))

    seed=config["seed"]
    if seed is not None:
        ml_utils.seed_all(seed)

    x_tr,y_tr,l_tr,x_te,y_te,l_te=utils.generate_data(config)
    x_tr_gen,y_tr_gen,l_tr_gen,x_te_gen,y_te_gen,l_te_gen=utils.generate_data(config,forgen=True)
    dataset_tr=torch.utils.data.TensorDataset(torch.tensor(x_tr).float(),torch.tensor(y_tr).float())
    dataset_te=torch.utils.data.TensorDataset(torch.tensor(x_te).float(),torch.tensor(y_te).float())
    
    utils.draw_setup(config,x_tr,x_te,os.path.join(experiment_directory,"setup.png"),l_tr=l_tr,l_te=l_te)

    n=len(dataset_tr)
    n_tr=int(n*config["train_ratio"])
    n_val=n-n_tr
    dataset_tr_tr,dataset_tr_val=torch.utils.data.random_split(dataset_tr,[n_tr,n_val])
    #print(dataset_tr_tr[0])

    model=utils.get_model(config)
    model=model.to(device)
    #print(model)

    batch_size=config["batch_size"]
    tr_loader=torch.utils.data.DataLoader(dataset_tr_tr,batch_size=batch_size,shuffle=True,num_workers=4)
    val_loader=torch.utils.data.DataLoader(dataset_tr_val,batch_size=batch_size,shuffle=False,num_workers=4)
    te_loader=torch.utils.data.DataLoader(dataset_te,batch_size=batch_size,shuffle=False,num_workers=4)

    ckpt_fol=os.path.join(experiment_directory,f"ckpts")
    os.makedirs(ckpt_fol)
    figure_fol=os.path.join(experiment_directory,f"figures")
    os.makedirs(figure_fol)
    generation_fol=os.path.join(experiment_directory,f"generations")
    os.makedirs(generation_fol)

    unconditioned=config.get("unconditioned",False)

    def batch_to_kwargs(batch):
        x,c=batch
        x=x.to(device)
        c=c.to(device)
        if unconditioned:
            c[:]=0.
        return {"x":x,"c":c}

    save_steps=[]
    val_losses=[]
    te_losses=[]
    ckpt_paths=[]
    def append_save_step(step,model):
        model.eval()
        save_steps.append(step)
    def append_val_te_loss(step,model):
        model.eval()
        with torch.no_grad():
            val_loss=0
            for batch in val_loader:
                loss=model.get_loss(**batch_to_kwargs(batch))
                val_loss+=loss.item()
            val_loss/=len(val_loader)
            te_loss=0
            for batch in te_loader:
                loss=model.get_loss(**batch_to_kwargs(batch))
                te_loss+=loss.item()
            te_loss/=len(te_loader)
        val_losses.append(val_loss)
        te_losses.append(te_loss)
    def save_ckpt(step,model):
        model.eval()
        ckpt_path=os.path.join(ckpt_fol,f"ckpt_step={step}.pth")
        torch.save(model.state_dict(),ckpt_path)
        ckpt_paths.append(ckpt_path)
    def generate(step,model):
        model.eval()
        c_tr=torch.tensor(y_tr_gen).float().to(device)
        c_te=torch.tensor(y_te_gen).float().to(device)
        if unconditioned:
            c_tr[:]=0.
            c_te[:]=0.
        gen_tr=model.generate(c=c_tr).detach().cpu().numpy()
        gen_te=model.generate(c=c_te).detach().cpu().numpy()
        gens={}
        gens["gen_tr"]=gen_tr
        gens["gen_te"]=gen_te
        gens["y_tr_gen"]=y_tr_gen
        gens["y_te_gen"]=y_te_gen
        gens["l_tr_gen"]=l_tr_gen
        gens["l_te_gen"]=l_te_gen
        generation_path=os.path.join(generation_fol,f"gen_{step}.pth")
        torch.save(gens,generation_path)
        figure_path=os.path.join(figure_fol,f"gen_{step}.png")
        utils.draw_setup(config,gen_tr,gen_te,figure_path,l_tr=l_tr_gen,l_te=l_te_gen)
    if config.get("save_ckpts",True):
        callbacks=[append_save_step,append_val_te_loss,save_ckpt,generate]
    else:
        callbacks=[append_save_step,append_val_te_loss,generate]

    train_results=ml_utils.train(model=model,dl_tr=tr_loader,batch_to_kwargs=batch_to_kwargs,
    n_steps=config["num_steps"],callback_steps=config["save_steps"],callbacks=callbacks,device=device)
    time_tr=train_results["time_tr"]
    time_callbacks=train_results["time_callbacks"]
    t_tr_r=time_tr/(time_tr+time_callbacks)*100
    t_cb_r=time_callbacks/(time_tr+time_callbacks)*100
    print(f"Time Split: Train:{t_tr_r:.2f}% Callback:{t_cb_r:.2f}%")

    logs={}
    logs.update(train_results)
    if hasattr(model,"data_noise"):#variational diffusion model
        logs["min_vlb"]=model.get_min_vlb()
    logs["losses"]=np.array(train_results["online_losses"])
    logs["save_steps"]=np.array(save_steps)
    logs["val_losses"]=np.array(val_losses)
    logs["te_losses"]=np.array(te_losses)
    logs["ckpt_paths"]=ckpt_paths
    torch.save(logs,os.path.join(experiment_directory,"logs.pth"))

    utils.plot_losses(logs,os.path.join(experiment_directory,"losses.png"))

    model.eval()