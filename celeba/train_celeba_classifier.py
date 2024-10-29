import os
os.chdir("/n/home12/cfpark00/ML/ToyCompDiff")

import PIL.Image
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

###
#setting 1

#save_path="/n/home12/cfpark00/ML/ToyCompDiff/data/celeba/run1_classifier"
#test=False

#attr1_name="Black_Hair"
#attr2_name='Eyeglasses'


####
#setting 2

#save_path="/n/home12/cfpark00/ML/ToyCompDiff/data/celeba/run2_classifier"
#test=False

#cnn_ckpt_path="./data/celeba/run1_classifier/ckpts/ckpt_10000.pth"

#attr1_name="Male"
#attr2_name="Smiling"

####
#setting 3

save_path="/n/home12/cfpark00/ML/ToyCompDiff/data/celeba/mh_classifier"
test=False

cnn_ckpt_path="./data/celeba/run1_classifier/ckpts/ckpt_10000.pth"

attr1_name="Male"
attr2_name="Wearing_Hat"

#####
if os.path.exists(save_path):
    assert False, "Path already exists"
os.makedirs(save_path, exist_ok=True)
ckpts_fol=os.path.join(save_path,"ckpts")
os.makedirs(ckpts_fol, exist_ok=True)

##
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


files_selected={}
vectors_selected={}
#n_per={(0,0):0,(0,1):0,(1,0):0,(1,1):0}
for key in files.keys():
    if test:
        files_selected[key]=PIL.Image.open(files[key])
    else:
        files_selected[key]=files[key]
    attributes=attributess[key]
    i1,i2=attributes[[i_h1,i_h2]]
    #n_per[(i1,i2)]+=1
    vectors_selected[key]=torch.tensor(i1*2+i2).to(torch.int64)
    if test and len(files_selected)>=100:
        break

dataset=Dataset(files_selected, vectors_selected)
n_tot=len(dataset)
print("Dataset size:",n_tot)
n_tr=int(n_tot*0.9)
n_val=n_tot-n_tr
ds_tr,ds_val=torch.utils.data.random_split(dataset,[n_tr,n_val])

batch_size=64
dl_tr=torch.utils.data.DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=True,num_workers=6)
dl_val=torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False,num_workers=6)

net=networks.CUNet(shape=(3,64,64),chs=[64,64,128],out_channels=64,mid_attn=True,num_res_blocks=2,norm_groups=4,)
classifier=models.Classifier(net=net,n_classes=[2,2])
classifier.optimizer=torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-5)
if cnn_ckpt_path is not None:
    cnn_ckpt=torch.load(cnn_ckpt_path)
    #delete weights not starting with "net"
    for k in list(cnn_ckpt.keys()):
        if not k.startswith("net"):
            del cnn_ckpt[k]
    classifier.load_state_dict(cnn_ckpt, strict=False)
    print("Loaded CNN ckpt")

classifier=classifier.to(device)

t_vals=[]
val_losses=[]
val_accs=[]
def batch_to_kwargs(batch):
    x,l=batch
    x=x.to(device)
    l=l.to(device)
    return {"x":x,"l":l}
def get_val_loss_acc(step,model):
    t_vals.append(step)
    model.eval()
    val_loss=0
    hits=[0,0]
    counts=0
    for batch in dl_val:
        kwargs=batch_to_kwargs(batch)
        loss=model.get_loss(**kwargs)
        prediction=model.classify(kwargs["x"])
        answer=model.unravel_index(kwargs["l"])
        for i in range(2):
            hits[i]+=(prediction[i]==answer[i]).sum().item()
        counts+=len(prediction[0])
        val_loss+=loss.item()
    val_loss/=len(dl_val)
    val_losses.append(val_loss)
    hits=[h/counts for h in hits]
    val_accs.append(hits)
    print(f"Step {step}, Val Loss: {val_loss}, Val Accs: {hits}")

def save_ckpt(step,model):
    torch.save(model.state_dict(),os.path.join(ckpts_fol,f"ckpt_{step}.pth"))
if test:
    n_steps=100
    callback_steps=[1,10,100]
else:
    n_steps=200_000
    callback_steps=[1,10,100,1000,3000,10_000,30_000,100_000,200_000]
callbacks=[save_ckpt,get_val_loss_acc]

train_results=ml_utils.train(model=classifier,dl_tr=dl_tr,
device=device,callback_steps=callback_steps,callbacks=callbacks,n_steps=n_steps,batch_to_kwargs=batch_to_kwargs)
train_results["t_vals"]=t_vals
train_results["val_losses"]=val_losses
train_results["val_accs"]=val_accs

torch.save(train_results, os.path.join(save_path,"train_results.pth"))

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_results["online_losses"])
plt.plot(t_vals,val_losses)
plt.yscale("log")
plt.xscale("log")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.subplot(1,2,2)
val_accs=np.array(val_accs)
plt.plot(t_vals,val_accs[:,0],label="Attribute 1")
plt.plot(t_vals,val_accs[:,1],label="Attribute 2")
plt.xlabel("Step")
plt.ylabel("Accuracy")
plt.xscale("log")
plt.legend()
plt.savefig(os.path.join(save_path,"losses.png"))

print("Done!!")