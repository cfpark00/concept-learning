import os
os.chdir("/n/home12/cfpark00/ML/ToyCompDiff")

import argparse
import utils
import numpy as np
import os

parser=argparse.ArgumentParser()
parser.add_argument("config_path",type=str,default="")
parser.add_argument("--n",default=128,type=int,nargs="+")
args=parser.parse_args()
yaml_path=args.config_path
n=args.n

n_classes_mult=[2,2,2]

config=utils.load_config(yaml_path)
n_classes=config["data_params"]["n_classes"]
assert n_classes==np.prod(n_classes_mult)
config["n_samples_train"]=[0 for _ in range(n_classes)]
config["n_samples_test"]=[0 for _ in range(n_classes)]
config["n_samples_train_gen"]=[0 for _ in range(n_classes)]
config["n_samples_test_gen"]=[n for _ in range(n_classes)]
_,_,_,x_te,_,l_te=utils.generate_data(config,seed=42,forgen=True)

"""
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
for l in range(n_classes):
    plt.subplot(2,4,l+1)
    plt.imshow(x_te[l_te==l][0].transpose(1,2,0))
    plt.axis("off")
plt.show()
"""
mses=np.zeros((n_classes,n_classes))
for l1 in range(n_classes):
    for l2 in range(n_classes):
        x1=x_te[l_te==l1]
        x2=x_te[l_te==l2]
        mse_pix=np.mean((x1-x2)**2)
        mses[l1,l2]=mse_pix
fol=config["experiment_directory"]
np.save(os.path.join(fol,"mse_pix.npy"),mses)


