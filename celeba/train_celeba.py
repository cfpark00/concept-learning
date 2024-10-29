import os
os.chdir("/n/home12/cfpark00/ML/ToyCompDiff")

import numpy as np
import torch
import matplotlib.pyplot as plt
import glob
import os
import PIL
import torchvision.transforms as transforms
import tqdm

import models
import utils

from mltools.networks import networks
from mltools.utils import cuda_tools
from mltools import ml_utils

device=cuda_tools.get_freer_device()

import sys
choice=int(sys.argv[1])

###
#setting 1

if choice==0:#hair and glasses
    save_path="/n/home12/cfpark00/ML/ToyCompDiff/data/celeba/run1"
    keys_file="/n/home12/cfpark00/ML/ToyCompDiff/data/celeba/selected1.pth"
    test=False
    attr1_name="Black_Hair"
    attr2_name='Eyeglasses'
    avoid=(1,1)
    seed=0
    ckpt_path=None
elif choice==1:#male and smiling
    save_path="/n/home12/cfpark00/ML/ToyCompDiff/data/celeba/run2"
    keys_file="/n/home12/cfpark00/ML/ToyCompDiff/data/celeba/selected2.pth"
    test=False
    attr1_name="Male"
    attr2_name="Smiling"
    avoid=(0,1)
    ckpt_path=None
elif choice==2:#male and smiling 2
    save_path="/n/home12/cfpark00/ML/ToyCompDiff/data/celeba/ms_bench_0"
    keys_file="/n/home12/cfpark00/ML/ToyCompDiff/data/celeba/selected2.pth"#intentional
    test=False
    attr1_name="Male"
    attr2_name="Smiling"
    avoid=(0,1)
    seed=0
    ckpt_path=None
elif choice==3:#male and smiling 3
    save_path="/n/home12/cfpark00/ML/ToyCompDiff/data/celeba/ms_bench_1"
    keys_file="/n/home12/cfpark00/ML/ToyCompDiff/data/celeba/selected2.pth"#intentional
    test=False
    attr1_name="Male"
    attr2_name="Smiling"
    avoid=(0,1)
    seed=100
    ckpt_path=None
elif choice==4:
    save_path="/n/home12/cfpark00/ML/ToyCompDiff/data/celeba/ms_bench_2"
    keys_file="/n/home12/cfpark00/ML/ToyCompDiff/data/celeba/selected2.pth"#intentional
    test=False
    attr1_name="Male"
    attr2_name="Smiling"
    avoid=(0,1)
    seed=200
    ckpt_path=None
elif choice==5:#rerun 2
    save_path="/n/home12/cfpark00/ML/ToyCompDiff/data/celeba/run2_re"
    keys_file="/n/home12/cfpark00/ML/ToyCompDiff/data/celeba/selected2.pth"#intentional
    test=False
    attr1_name="Male"
    attr2_name="Smiling"
    avoid=(0,1)
    ckpt_path="/n/home12/cfpark00/ML/ToyCompDiff/data/celeba/run2/ckpts/ckpt_1000000.pth"
    seed=0
elif choice==6:#rerun 2
    save_path="/n/home12/cfpark00/ML/ToyCompDiff/data/celeba/mh_bench_0"
    keys_file="/n/home12/cfpark00/ML/ToyCompDiff/data/celeba/selected3.pth"#intentional
    test=False
    attr1_name="Male"
    attr2_name="Wearing_Hat"
    avoid=(0,1)
    ckpt_path=None
    seed=0
#####
if test:
    n_steps=100
    callback_steps=[1,10,100]
else:
    n_steps=2_000_000
    callback_steps=[    500,    2111,    4836,    8673,   13624,   19688,   26864,
         35154,   44557,   55073,   66702,   79444,   93299,  108267,
        124348,  141542,  159849,  179270,  199803,  221449,  244209,
        268081,  293067,  319165,  346377,  374701,  404139,  434690,
        466353,  499130,  533020,  568023,  604139,  641368,  679710,
        719165,  759733,  801414,  844208,  888115,  933136,  979269,
       1026515, 1074875, 1124347, 1174933, 1226631, 1279443, 1333368,
       1388405, 1444556, 1501820, 1560197, 1619686, 1680289, 1742005,
       1804834, 1868776, 1933831, 2000000]
ml_utils.seed_all(seed)

#####

####
if os.path.exists(save_path):
    assert False, "Path already exists"
ckpts_fol=os.path.join(save_path,"ckpts")
os.makedirs(ckpts_fol, exist_ok=True)
generations_fol=os.path.join(save_path,"generations")
os.makedirs(generations_fol, exist_ok=True)
imgs_fol=os.path.join(save_path,"imgs")
os.makedirs(imgs_fol, exist_ok=True)


#####

csvpath="/n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/ML/datasets/celebA/list_attr_celeba.txt"
imgdir="/n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/ML/datasets/celebA/imgs"
files_=glob.glob(os.path.join(imgdir, "*.png"))
files={}
for f in files_:
    name=os.path.basename(f)
    files[name.replace(".png","")]=f
header=list(np.loadtxt(csvpath, skiprows=1, max_rows=1, dtype=str))
attributess_=np.loadtxt(csvpath, skiprows=2, dtype=str)
attributess={}
for i, attrs in enumerate(attributess_):
    key=attrs[0].replace(".jpg","")
    attributess[key]=np.array([1 if x=="1" else 0 for x in attrs[1:]]).astype(np.int64)

i_h1=header.index(attr1_name)
i_h2=header.index(attr2_name)

beta_settings={"gamma_max":10.0, "gamma_min":-5.0, "noise_schedule":"learned_linear", "type":"logsnr"}
net=networks.CUNet(shape=(3,64,64),chs=[64,128,256],mid_attn=True,num_res_blocks=2,norm_groups=4,v_conditioning_dims=[4],t_conditioning=True)
vdm=models.GenVDiff(net=net,beta_settings=beta_settings,data_noise=1e-3)
if ckpt_path is not None:
    vdm.load_state_dict(torch.load(ckpt_path))
    print("Loaded model from",ckpt_path)
vdm.optimizer=torch.optim.AdamW(vdm.parameters(), lr=1e-3, weight_decay=1e-5)
vdm=vdm.to(device)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, files_dict, vectors_dict):
        self.keys=list(files_dict.keys())
        self.files_dict=files_dict
        self.vectors_dict=vectors_dict
        #add translation and color small gitter
        self.trf=transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor(), transforms.ColorJitter(0.1,0.1,0.1,0.1), transforms.RandomHorizontalFlip()])
    def __getitem__(self, idx):
        key=self.keys[idx]
        file=self.files_dict[key]
        if isinstance(file, str):
            img=PIL.Image.open(file)
        elif isinstance(file, PIL.Image.Image):
            img=file
        else:
            raise ValueError("Unsupported file type")
        img=self.trf(img)
        vector=self.vectors_dict[key]
        return img, vector
    def __len__(self):
        return len(self.keys)


keyss=torch.load(keys_file)
tot=sum([len(keys) for keys in keyss.values()])
files_selected={}
vectors_selected={}
examples={}
with tqdm.tqdm(total=tot,desc="Loading Images...") as pbar:
    for keys in keyss.values():
        count=0
        for key in keys:
            attributes=attributess[key]
            iden=attributes[[i_h1,i_h2]]
            idenkey=tuple(iden)
            if iden[0]==avoid[0] and iden[1]==avoid[1]:
                assert False, str(avoid)+" should not be in the dataset"
            else:
                files_selected[key]=PIL.Image.open(files[key])
                attrs=torch.tensor(iden)
                vectors_selected[key]=torch.nn.functional.one_hot(attrs,2).float().view(-1)
                if idenkey not in examples:
                    examples[idenkey]=files_selected[key]
                pbar.update(1)
                count+=1
            if test and count>=250:
                break
plt.figure(figsize=(10,10))
for i, (idenkey, img) in enumerate(examples.items()):
    plt.subplot(2,2,i+1)
    plt.imshow(img)
    plt.title(str(idenkey))
plt.savefig(os.path.join(save_path,"examples.png"))

dataset=Dataset(files_selected, vectors_selected)
print("Dataset size:",len(dataset))
batch_size=64
dl_tr=torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,num_workers=12)

def batch_to_kwargs(batch):
    x,c=batch
    x=x.to(device)
    c=c.to(device)
    return {"x":x,"c":c}
def save_ckpt(step,model):
    torch.save(model.state_dict(),os.path.join(ckpts_fol,f"ckpt_{step}.pth"))
def generate(step,model):
    model.eval()
    batch=32
    gens={}
    for l in [(0,0),(0,1),(1,0),(1,1)]:
        l_=torch.tensor(l).to(torch.int64)
        c=torch.nn.functional.one_hot(l_,2).float().view(1,-1).repeat(batch,1).to(device)#shape (32,4)
        assert c.shape==(32,4)
        gen=model.generate(c=c).detach().cpu()
        img=np.clip(gen[0].permute(1,2,0).numpy(),0,1)
        plt.imsave(os.path.join(imgs_fol,f"gen_{step}_l={l}.png"),img)
        gens[l]=gen
    torch.save(gens,os.path.join(generations_fol,f"gen_{step}.pth"))
callbacks=[save_ckpt,generate]

train_results=ml_utils.train(model=vdm,dl_tr=dl_tr,
device=device,callback_steps=callback_steps,callbacks=callbacks,n_steps=n_steps,batch_to_kwargs=batch_to_kwargs)

torch.save(train_results, os.path.join(save_path,"train_results.pth"))

try:
    min_vlb=vdm.get_min_vlb()
    print("Min VLB:",min_vlb)
except:
    min_vlb=0
plt.figure(figsize=(10,5))
plt.plot(train_results["online_losses"]-min_vlb)
plt.yscale("log")
plt.xscale("log")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.savefig(os.path.join(save_path,"losses.png"))

print("Done!!")