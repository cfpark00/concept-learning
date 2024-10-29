import numpy as np
import torch
import yaml
import scipy.ndimage as sim

from mltools.networks import networks

import models

def process_config(config,check_only=False):
    if "dataset" not in config or config["dataset"]=="vec":
        dim=config['dim']
        dim_nuisance=config['dim_nuisance']
        n_classes = config['n_classes']
        inds_md=np.array(config['inds_md'])
        means=np.array(config['means'])
        covs=np.array(config['covs'])
        noise_covs=np.array(config['noise_covs'])
        if "n_samples_train" in config:
            assert "n_samples_test" in config
            n_samples_train=np.array(config['n_samples_train'])
            n_samples_test=np.array(config['n_samples_test'])
        else:
            assert "n_samples" in config
            n_samples=np.array(config['n_samples'])
            n_samples_train=n_samples
            n_samples_test=n_samples
        batch_size=config['batch_size']
        #train_inds=np.array(config['train_inds'])
        #if "test_inds" in config:
        #    test_inds=np.array(config['test_inds'])
        #else:
        #    test_inds=np.arange(n_classes)

        fig_x=np.array(config['fig_x'])
        fig_y=np.array(config['fig_y'])

        assert inds_md.shape==(n_classes,dim)
        assert means.shape==(n_classes,dim)
        assert covs.shape==(n_classes,dim,dim)
        assert noise_covs.shape==(n_classes,dim,dim)
        assert n_samples_train.shape==(n_classes,)
        assert n_samples_test.shape==(n_classes,)
        #assert train_inds.min()>=0 and train_inds.max()<n_classes
        #assert test_inds.min()>=0 and test_inds.max()<n_classes
        assert fig_x.shape[-1]==(dim+dim_nuisance) and fig_y.shape[-1]==(dim+dim_nuisance)
        assert all([n%batch_size==0 for n in n_samples_train])
        assert all([n%batch_size==0 for n in n_samples_test])

        if not check_only:
            config['n_classes']=n_classes
            config['inds_md']=inds_md
            config['means']=means
            global_mean=np.mean(means,axis=0)
            global_std=np.std(means,axis=0)
            config['global_mean']=global_mean
            config['global_std']=global_std
            config['covs']=covs
            config['noise_covs']=noise_covs
            config['n_samples_train']=n_samples_train
            config['n_samples_test']=n_samples_test
            #config['train_inds']=train_inds
            #config['test_inds']=test_inds
            config['fig_x']=fig_x
            config['fig_y']=fig_y

            num_steps=config['num_steps']
            save_steps=config['save_steps']
            if isinstance(save_steps,int):
                save_steps=(np.linspace(np.sqrt(10),np.sqrt(num_steps),save_steps)**2).astype(int)
                save_steps[-1]=num_steps
                save_steps=np.unique(save_steps)
                config['save_steps']=save_steps
            elif isinstance(save_steps,list):
                save_steps=np.array(save_steps)
                config['save_steps']=save_steps
            else:
                raise ValueError("Invalid save_steps type")
    elif config["dataset"]=="images_1":
        data_params=config["data_params"]
        n_classes=data_params["n_classes"]
        comp_dims=data_params["comp_dims"]
        n_check=1
        for k,v in comp_dims.items():
            if v is not None:
                n_check*=v
        assert n_classes==n_check, f"n_classes does not match comp_dims {n_classes}!={n_check}"
        assert len(config["n_samples_train"])==n_classes
        assert len(config["n_samples_test"])==n_classes
        assert len(config["n_samples_train_gen"])==n_classes
        assert len(config["n_samples_test_gen"])==n_classes


        if not check_only:
            config["data_params"]["means"]=np.array(config["data_params"]["color"]["means"])
            config["data_params"]["means"]=np.array(config["data_params"]["size"]["means"])
            config["data_params"]["means"]=np.array(config["data_params"]["bg_color"]["means"])
            num_steps=config['num_steps']
            save_steps=config['save_steps']
            if isinstance(save_steps,int):
                save_steps_start=config.get("save_steps_start",10)
                save_steps=(np.linspace(np.sqrt(save_steps_start),np.sqrt(num_steps),save_steps)**2).astype(int)
                save_steps[-1]=num_steps
                save_steps=np.unique(save_steps)
                config['save_steps']=save_steps
            elif isinstance(save_steps,list):
                save_steps=np.array(save_steps)
                config['save_steps']=save_steps
            else:
                raise ValueError("Invalid save_steps type")
    return config

def load_config(yaml_path):
    config=yaml.safe_load(open(yaml_path))
    config=process_config(config)
    return config

def sigmoid(x):
    return 1/(1+np.exp(-x))

def one_hot(l,n_class):
    return np.eye(n_class)[l]

def multi_one_hot(l,perdim):
    dim=len(l)
    y=[]
    for i in range(dim):
        y.append(one_hot(l[i],perdim[i]))
    return np.concatenate(y,axis=0).astype(np.float32)

def generate_data(config,forgen=False,**kwargs):
    if "dataset" not in config or config["dataset"]=="vec":
        return generate_data_vec(config,forgen=forgen,**kwargs)
    elif config["dataset"]=="images_1":
        return generate_data_images_1(config,forgen=forgen,**kwargs)
    else:
        raise NotImplementedError(f"Dataset {config['dataset']} not implemented")

def generate_data_vec(config,forgen=False):
    dim=config['dim']
    dim_nuisance=config.get("dim_nuisance",0)
    nuisance_scale=config.get("nuisance_scale",1.)
    n_classes=config['n_classes']
    means=config['means']
    covs=config['covs']
    noise_covs=config['noise_covs']
    inds_md=config["inds_md"]
    if forgen:
        n_samples_train=config["n_samples_train_gen"]
        n_samples_test=config["n_samples_test_gen"]
    else:
        n_samples_train=config['n_samples_train']
        n_samples_test=config['n_samples_test']
    randmat=config.get("randmat",np.eye(dim))
    sigmoidy=config.get("sigmoidy",False)
    ymean=config.get("ymean",False)
    classy=config.get("classy",False)
    if classy:
        perdim=np.array(config["perdim"])
        assert n_classes==int(np.prod(perdim))

    x_trs=[]
    y_trs=[]
    l_trs=[]
    x_tes=[]
    y_tes=[]
    l_tes=[]
    for i in range(n_classes):
        mean=means[i]
        cov=covs[i]
        noise_cov=noise_covs[i]

        def gen(n_sample):
            x=np.random.multivariate_normal(mean,cov,n_sample)
            l=np.full(n_sample,i)
            if classy:
                ind_md=inds_md[i]
                y=multi_one_hot(ind_md,perdim)
                y=y[None,:].repeat(n_sample,axis=0)
            else:
                if ymean:
                    y=mean[None,:].repeat(n_sample,axis=0)
                else:
                    n=np.random.multivariate_normal(np.zeros(dim),noise_cov,n_sample)
                    y=x@randmat+n
                    if sigmoidy:
                        y=sigmoid(y)

            if dim_nuisance>0:
                if "nl_nuisance" in config and config["nl_nuisance"]:
                    assert n_sample%(2**dim_nuisance)==0
                    arr=np.array([-1.,1.])
                    nuisance_ds=np.stack(np.meshgrid(*[arr]*dim_nuisance,indexing="ij"),axis=-1)
                    nuisance_ds=nuisance_ds.reshape(-1,dim_nuisance)#2**dim_nuisance,dim_nuisance
                    n_rep=n_sample//(2**dim_nuisance)
                    nuisance_ds=nuisance_ds[None,:].repeat(n_rep,axis=0).reshape(-1,dim_nuisance)
                    nuisance_s=0.1
                    nuisance=np.random.randn(n_sample,dim_nuisance)*nuisance_s+nuisance_ds
                    x=np.concatenate([x,nuisance],axis=-1)
                else:
                    x=np.concatenate([x,np.random.randn(n_sample,dim_nuisance)*nuisance_scale],axis=-1)
            return x,y,l
        
        n_sample=n_samples_train[i]
        if n_sample!=0:
            x,y,l=gen(n_sample)
            x_trs.append(x)
            y_trs.append(y)
            l_trs.append(l)

        n_sample=n_samples_test[i]
        if n_sample!=0:
            x,y,l=gen(n_sample)
            x_tes.append(x)
            y_tes.append(y)
            l_tes.append(l)
    n_tr=len(x_trs)
    n_te=len(x_tes)
    if n_tr>0:
        x_tr=np.concatenate(x_trs,axis=0)
        y_tr=np.concatenate(y_trs,axis=0)
        l_tr=np.concatenate(l_trs,axis=0)
    else:
        x_tr,y_tr,l_tr=[],[],[]
    if n_te>0:
        x_te=np.concatenate(x_tes,axis=0)
        y_te=np.concatenate(y_tes,axis=0)
        l_te=np.concatenate(l_tes,axis=0)
    else:
        x_te,y_te,l_te=[],[],[]
    return x_tr,y_tr,l_tr,x_te,y_te,l_te

def get_comp_classes_images_1(i_class,config):
    comp_dims=config["data_params"]["comp_dims"]
    comp_ns=[]
    comp_names=[]
    for k in ["shape","x","y","color","size","bg_color"]:
        dim=comp_dims[k]
        if dim is not None:
            comp_ns.append(dim)
            comp_names.append(k)
    comp_classes_=np.unravel_index(i_class,comp_ns)
    comp_classes={}
    for k,comp_class in zip(comp_names,comp_classes_):
        comp_classes[k]=comp_class
    return comp_classes

def generate_images_vecss(**kwargs):
    image_size=kwargs.get("image_size",32)
    n_sample=kwargs.get("n_sample",128)
    noise_level=kwargs.get("noise_level",0.001)
    shape_name=kwargs.get("shape_name","circle")
    x_means=kwargs.get("x_means",np.random.uniform(-1,1,n_sample))
    y_means=kwargs.get("y_means",np.random.uniform(-1,1,n_sample))
    colors=kwargs.get("colors",np.random.uniform(0,1,(n_sample,3)))
    sizes=kwargs.get("sizes",np.random.uniform(0,1,n_sample))
    bg_colors=kwargs.get("bg_colors",np.random.uniform(0,1,(n_sample,3)))
    x_s_n=kwargs.get("x_s_n",0.0)
    y_s_n=kwargs.get("y_s_n",0.0)
    color_s_n=kwargs.get("color_s_n",0.0)
    size_s_n=kwargs.get("size_s_n",0.0)
    bg_color_s_n=kwargs.get("bg_color_s_n",0.0)
    comp_classes=kwargs.get("comp_classes",{"color":0,"size":0})


    arr=np.linspace(-1.,1.,image_size)
    x_grid,y_grid=np.meshgrid(arr,arr,indexing="ij")
    images=[]
    vecss=[]
    for i in range(n_sample):
        image=np.zeros((image_size,image_size,3),dtype=np.float32)
        x=x_means[i]
        y=y_means[i]
        color=colors[i]
        size=sizes[i]
        bg_color=bg_colors[i]

        dx=x_grid-x
        dy=y_grid-y
        if shape_name=="circle":
            r=np.sqrt(dx**2+dy**2)
            mask=r<size
        elif shape_name=="triangle":
            triangle_side=np.sqrt(4*np.pi/np.sqrt(3))*size
            incircle=triangle_side*(np.sqrt(3)/6)
            a1,b1=2,(1-(np.sqrt(3)/6))*triangle_side
            a2,b2=-2,(1-(np.sqrt(3)/6))*triangle_side
            mask=(dy>(-incircle))*(dy<(a1*dx+b1))*(dy<(a2*dx+b2))
        smoothmask=sim.gaussian_filter(mask.astype(np.float32),1.)
        image+=smoothmask[:,:,None]*color[None,None,:]+(1-smoothmask[:,:,None])*bg_color[None,None,:]
        noise=np.random.randn(image_size,image_size,3)*noise_level
        image+=noise
        image=np.clip(image,0,1).astype(np.float32)

        vecs=[]
        for key in ["shape","x","y","color","size","bg_color"]:
            if key=="shape":
                if shape_name=="circle":
                    vec=np.array([1.,0.])
                elif shape_name=="triangle":
                    vec=np.array([0.,1.])
            elif key=="x":
                vec=x+np.random.randn(1)*x_s_n
            elif key=="y":
                vec=y+np.random.randn(1)*y_s_n
            elif key=="color":
                vec=color+np.random.randn(3)*color_s_n
            elif key=="size":
                vec=size+np.random.randn(1)*size_s_n
            elif key=="bg_color":
                vec=bg_color+np.random.randn(3)*bg_color_s_n
            else:
                raise NotImplementedError(f"Key {key} not implemented")

            if key in comp_classes:
                vecs.append(vec)
            else:
                vecs.append(np.zeros_like(vec))
        vecs=np.concatenate(vecs,axis=0)
            
        images.append(image)
        vecss.append(vecs)
    images=np.stack(images,axis=0)
    vecss=np.stack(vecss,axis=0)
    return images,vecss

def generate_images_1(i_class,n_sample,config,test=False):
    comp_classes=get_comp_classes_images_1(i_class,config)
    ###
    image_size=config["data_params"]["image_size"]
    noise_level=config["data_params"]["noise_level"]

    #dim -6: shape
    if "shape" not in config["data_params"]:
        shape_name="circle"
    else:
        #print(comp_classes["shape"])
        if "shape" not in comp_classes:
            shape_name="circle"
        else:
            shape_name=config["data_params"]["shape"]["names"][comp_classes["shape"]]

    #dim -5: x
    x_min=config["data_params"]["x"]["min"]
    x_max=config["data_params"]["x"]["max"]
    x_n=config["data_params"]["x"]["n"]
    x_s=config["data_params"]["x"]["s"]
    x_s_n=config["data_params"]["x"]["s_n"]
    if x_n is None:
        x_means=np.random.uniform(x_min,x_max,n_sample)
    else:
        x_means=np.full(n_sample,np.linspace(x_min,x_max,x_n)[comp_classes["x"]])
        if not test:
            x_means+=np.random.randn(n_sample)*x_s

    #dim -4: y
    y_min=config["data_params"]["y"]["min"]
    y_max=config["data_params"]["y"]["max"]
    y_n=config["data_params"]["y"]["n"]
    y_s=config["data_params"]["y"]["s"]
    y_s_n=config["data_params"]["y"]["s_n"]
    if y_n is None:
        y_means=np.random.uniform(y_min,y_max,n_sample)
    else:
        y_means=np.full(n_sample,np.linspace(y_min,y_max,y_n)[comp_classes["y"]])
        if not test:
            y_means+=np.random.randn(n_sample)*y_s

    #dim -3: color
    color_s_n=config["data_params"]["color"]["s_n"]
    color_mean=np.array(config["data_params"]["color"]["means"][comp_classes["color"]])
    color_min=np.array(config["data_params"]["color"]["mins"][comp_classes["color"]])
    color_max=np.array(config["data_params"]["color"]["maxs"][comp_classes["color"]])
    color_range=color_max-color_min
    if test:
        colors=np.full((n_sample,3),color_mean)
    else:
        colors=np.random.uniform(0,1,(n_sample,3)).astype(np.float32)*color_range[None,:]+color_min[None,:]

    #dim -2: object size
    size_s_n=config["data_params"]["size"]["s_n"]
    size_mean=config["data_params"]["size"]["means"][comp_classes["size"]]
    size_min=config["data_params"]["size"]["mins"][comp_classes["size"]]
    size_max=config["data_params"]["size"]["maxs"][comp_classes["size"]]
    size_min_=config["data_params"]["size"]["min"]
    size_min=np.maximum(size_min,size_min_)
    size_range=size_max-size_min
    if test:
        sizes=np.full(n_sample,size_mean)
    else:
        sizes=np.random.uniform(0,1,n_sample)*size_range+size_min

    #dim -1: background color
    bg_color_s_n=config["data_params"]["bg_color"]["s_n"]
    if "bg_color" not in comp_classes:
        bg_color_mean=np.array(config["data_params"]["bg_color"]["means"][0])
        bg_color_min=np.array(config["data_params"]["bg_color"]["mins"][0])
        bg_color_max=np.array(config["data_params"]["bg_color"]["maxs"][0])
    else:
        bg_color_mean=np.array(config["data_params"]["bg_color"]["means"][comp_classes["bg_color"]])
        bg_color_min=np.array(config["data_params"]["bg_color"]["mins"][comp_classes["bg_color"]])
        bg_color_max=np.array(config["data_params"]["bg_color"]["maxs"][comp_classes["bg_color"]])
    bg_color_range=bg_color_max-bg_color_min
    if test:
        bg_colors=np.full((n_sample,3),bg_color_mean)
    else:
        bg_colors=np.random.uniform(0,1,(n_sample,3)).astype(np.float32)*bg_color_range[None,:]+bg_color_min[None,:]

    arr=np.linspace(-1.,1.,image_size)
    x_grid,y_grid=np.meshgrid(arr,arr,indexing="ij")
    images=[]
    vecss=[]
    for i in range(n_sample):
        image=np.zeros((image_size,image_size,3),dtype=np.float32)
        x=x_means[i]
        y=y_means[i]
        color=colors[i]
        size=sizes[i]
        bg_color=bg_colors[i]

        dx=x_grid-x
        dy=y_grid-y
        if shape_name=="circle":
            r=np.sqrt(dx**2+dy**2)
            mask=r<size
        elif shape_name=="triangle":
            triangle_side=np.sqrt(4*np.pi/np.sqrt(3))*size
            incircle=triangle_side*(np.sqrt(3)/6)
            a1,b1=2,(1-(np.sqrt(3)/6))*triangle_side
            a2,b2=-2,(1-(np.sqrt(3)/6))*triangle_side
            mask=(dy>(-incircle))*(dy<(a1*dx+b1))*(dy<(a2*dx+b2))
        smoothmask=sim.gaussian_filter(mask.astype(np.float32),1.)
        image+=smoothmask[:,:,None]*color[None,None,:]+(1-smoothmask[:,:,None])*bg_color[None,None,:]
        noise=np.random.randn(image_size,image_size,3)*noise_level
        image+=noise
        image=np.clip(image,0,1).astype(np.float32)

        vecs=[]
        for key in ["shape","x","y","color","size","bg_color"]:
            if key=="shape":
                if shape_name=="circle":
                    vec=np.array([1.,0.])
                elif shape_name=="triangle":
                    vec=np.array([0.,1.])
            elif key=="x":
                vec=x+np.random.randn(1)*x_s_n
            elif key=="y":
                vec=y+np.random.randn(1)*y_s_n
            elif key=="color":
                vec=color+np.random.randn(3)*color_s_n
            elif key=="size":
                vec=size+np.random.randn(1)*size_s_n
            elif key=="bg_color":
                vec=bg_color+np.random.randn(3)*bg_color_s_n
            else:
                raise NotImplementedError(f"Key {key} not implemented")

            if key in comp_classes:
                vecs.append(vec)
            else:
                vecs.append(np.zeros_like(vec))
        vecs=np.concatenate(vecs,axis=0)
            
        images.append(image)
        vecss.append(vecs)
    images=np.stack(images,axis=0)
    vecss=np.stack(vecss,axis=0)
    return images,vecss,np.full(n_sample,i_class)

def generate_data_images_1(config,forgen=False,seed=None):
    n_classes=config["data_params"]["n_classes"]
    if forgen:
        n_samples_train=config["n_samples_train_gen"]
        n_samples_test=config["n_samples_test_gen"]
    else:
        n_samples_train=config['n_samples_train']
        n_samples_test=config['n_samples_test']
    x_trs=[]
    y_trs=[]
    l_trs=[]
    x_tes=[]
    y_tes=[]
    l_tes=[]
    for i_class in range(n_classes):
        if seed is not None:
            np.random.seed(seed)
        n_sample=n_samples_train[i_class]
        if n_sample!=0:
            x,y,l=generate_images_1(i_class,n_sample,config,test=False)
            x_trs.append(x)
            y_trs.append(y)
            l_trs.append(l)
        n_sample=n_samples_test[i_class]
        if n_sample!=0:
            x,y,l=generate_images_1(i_class,n_sample,config,test=True)
            x_tes.append(x)
            y_tes.append(y)
            l_tes.append(l)
    n_tr=len(x_trs)
    n_te=len(x_tes)
    if n_tr>0:
        x_tr=np.concatenate(x_trs,axis=0).transpose(0,3,1,2)
        y_tr=np.concatenate(y_trs,axis=0)
        l_tr=np.concatenate(l_trs,axis=0)
    else:
        x_tr,y_tr,l_tr=[],[],[]
    if n_te>0:
        x_te=np.concatenate(x_tes,axis=0).transpose(0,3,1,2)
        y_te=np.concatenate(y_tes,axis=0)
        l_te=np.concatenate(l_tes,axis=0)
    else:
        x_te,y_te,l_te=[],[],[]
    return x_tr,y_tr,l_tr,x_te,y_te,l_te
    
def get_classifier_images_1(ckpt_path=None):
    from mltools.networks import networks
    net=networks.CUNet(shape=(3,32,32),out_channels=64,chs=[32,32,32],norm_groups=4)
    classifier=models.Classifier(net=net,n_classes=[2,2,2])
    if ckpt_path is not None:
        classifier.load_state_dict(torch.load(ckpt_path))
    return classifier

import os
import glob
def get_ckpt_paths(fol):
    ckptfol=os.path.join(fol,"ckpts")
    def get_step(path):
        name=os.path.basename(path)
        return int(name.split(".")[0].split("step=")[-1])
    ckpt_paths=glob.glob(os.path.join(ckptfol,"*.pth"))
    steps=[get_step(path) for path in ckpt_paths]
    inds=np.argsort(steps)
    steps=[steps[i] for i in inds]
    ckpt_paths=[ckpt_paths[i] for i in inds]
    ckpt_paths=dict(zip(steps,ckpt_paths))
    return ckpt_paths

def get_generation_paths(fol,suffix=""):
    generationsfol=os.path.join(fol,"generations"+("" if suffix=="" else "_"+suffix))
    def get_step(path):
        name=os.path.basename(path)
        return int(name.split(".")[0].split("_")[-1])
    generation_paths=glob.glob(os.path.join(generationsfol,"*.pth"))
    steps=[get_step(path) for path in generation_paths]
    inds=np.argsort(steps)
    steps=[steps[i] for i in inds]
    generation_paths=[generation_paths[i] for i in inds]
    generation_paths=dict(zip(steps,generation_paths))
    return generation_paths
    

def get_model(config):
    model_params=config["model_params"]

    if "dataset" not in config or config["dataset"]=="vec":
        dim=config['dim']
        dim_nuisance=config.get("dim_nuisance",0)

        dim_x=dim+dim_nuisance
        if "classy" in config and config["classy"]:
            perdim=np.array(config["perdim"])
            dim_c=perdim.sum()
        else:
            dim_c=dim

        model_type=model_params["model_type"]
        network_type=model_params["network_type"]
        hidden_dims=model_params["hidden_dims"]
        init_scale=model_params["init_scale"]
        optimizer_type=model_params["optimizer_type"]
        optimizer_params=model_params["optimizer_params"]
        ckpt=model_params.get("ckpt",None)
        zero_bias=model_params.get("zero_bias",False)
        beta_settings=model_params.get("beta_settings",{"type":"logsnr","T":100,"logsnr_i":3,"logsnr_f":-3})

        cond_init_zero=model_params.get("cond_init_zero",False)

        if network_type=="MLP":
            net=networks.CMLP(in_dim=dim_x,h_dims=hidden_dims,
                            t_conditioning=True if "Diff" in model_params["model_type"] else False,
                            v_conditioning_dims=[dim_c])
            for n,p in net.named_parameters():
                p.data*=init_scale
                if n.endswith("bias") and zero_bias:
                    p.data.zero_()
                if cond_init_zero:
                    if "embedders" in n and ("2.weight" in n or "2.bias" in n):
                        p.data.zero_()
        
        if model_type=="Det":
            model=models.GenDet(net=net)
        elif model_type=="Diff":
            model=models.GenDiff(net=net,beta_settings=beta_settings)
        elif model_type=="VDiff":
            assert "data_noise" in model_params
            model=models.GenVDiff(net=net,beta_settings=beta_settings,data_noise=model_params["data_noise"])
        else:
            raise NotImplementedError(f"Model type {model_type} not implemented")
        optimizer=getattr(torch.optim,optimizer_type)(model.parameters(),**optimizer_params)
        model.optimizer=optimizer

        if ckpt is not None:
            model.load_state_dict(torch.load(ckpt))
        return model

    elif config["dataset"]=="images_1":
        image_size=config["data_params"]["image_size"]
        shape=(3,image_size,image_size)

        model_type=model_params["model_type"]
        network_params=model_params["network_params"]
        optimizer_type=model_params["optimizer_type"]
        optimizer_params=model_params["optimizer_params"]
        ckpt=model_params.get("ckpt",None)
        beta_settings=model_params["beta_settings"]

        net=networks.CUNet(shape=shape,**network_params,t_conditioning=True if "Diff" in model_params["model_type"] else False,)
        
        if model_type=="Det":
            model=models.GenDet(net=net)
        elif model_type=="Diff":
            model=models.GenDiff(net=net,beta_settings=beta_settings)
        elif model_type=="VDiff":
            assert "data_noise" in model_params
            model=models.GenVDiff(net=net,beta_settings=beta_settings,data_noise=model_params["data_noise"],
                                  p_cfg=model_params.get("p_cfg",None),w_cfg=model_params.get("w_cfg",None))
        else:
            raise NotImplementedError(f"Model type {model_type} not implemented")
        
        optimizer=getattr(torch.optim,optimizer_type)(model.parameters(),**optimizer_params)
        model.optimizer=optimizer

        if ckpt is not None:
            model.load_state_dict(torch.load(ckpt))
        return model
    else:
        raise NotImplementedError(f"Dataset {config['dataset']} not implemented")

import matplotlib.pyplot as plt
import matplotlib

def draw_setup(config,x_tr,x_te,save_path=None,l_tr=None,l_te=None):
    figsize_setup=(8,8)
    if config["dataset"]=="vec":
        fig_x=config["fig_x"]
        fig_y=config["fig_y"]
        if fig_x.ndim==1:
            fig_x=fig_x[None,:]
        if fig_y.ndim==1:
            fig_y=fig_y[None,:]
        
        n_plots=fig_x.shape[0]+1
        fig=plt.figure(figsize=(n_plots*figsize_setup[0],figsize_setup[1]))
        c=0
        for fig_x_,fig_y_ in zip(fig_x,fig_y):
            x_c_tr=x_tr@fig_x_
            y_c_tr=x_tr@fig_y_
            x_c_te=x_te@fig_x_
            y_c_te=x_te@fig_y_

            plt.subplot(1,n_plots,c+1)
            plt.scatter(x_c_tr,y_c_tr,c="black",label="Training Data",s=10)
            plt.scatter(x_c_te,y_c_te,c="red",label="C.G. Target",s=10)
            plt.legend()
            plt.title(f"X:{list(np.round(fig_x_,2))} vs Y:{list(np.round(fig_y_,2))}")
            c+=1
        plt.subplot(1,n_plots,c+1)
        texty=0
        if l_tr is not None:
            for l in np.unique(l_tr):
                inds=l_tr==l
                center=np.mean(x_tr[inds],axis=0)
                plt.annotate(f"Train {l}: {center}",xy=(0,texty),fontsize=20)
                texty-=1
        if l_te is not None:
            for l in np.unique(l_te):
                inds=l_te==l
                center=np.mean(x_te[inds],axis=0)
                plt.annotate(f"Test {l}: {center}",xy=(0,texty),fontsize=20)
                texty-=1
        plt.axis("off")
        plt.ylim(texty-1,1)
    elif config["dataset"]=="images_1":
        assert l_tr is not None and l_te is not None
        n_classes=config["data_params"]["n_classes"]
        n_col=config["fig_n_col"]
        n_rows=np.ceil(n_classes/n_col).astype(int)
        fig=plt.figure(figsize=(n_col*6,n_rows*6))
        c=0
        for i_class in range(n_classes):
            comp_classes=get_comp_classes_images_1(i_class,config)
            plt.subplot(n_rows,n_col,c+1)
            title=f"Class {i_class}"
            if i_class in l_tr:
                i_sel=np.random.choice(np.nonzero(l_tr==i_class)[0])
                x=np.clip(x_tr[i_sel].transpose(1,2,0),0,1)
                title+=" (Train)\n"+str(comp_classes)
            elif i_class in l_te:
                i_sel=np.random.choice(np.nonzero(l_te==i_class)[0])
                x=np.clip(x_te[i_sel].transpose(1,2,0),0,1)
                title+=" (Test)\n"+str(comp_classes)
            else:
                title+=" (No Data)"
            plt.imshow(x.transpose(1,0,2),origin="lower")
            plt.title(title)
            c+=1

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
        return
    return fig

def plot_losses(logs,save_path=None):
    figsize_loss=(8,5)
    fig=plt.figure(figsize=figsize_loss)
    if "min_vlb" in logs:
        offset=logs["min_vlb"]
    else:
        offset=0
    plt.plot(logs["losses"]-offset,label="Train Loss")
    plt.plot(logs["save_steps"],logs["val_losses"]-offset,label="Val Loss")
    plt.plot(logs["save_steps"],logs["te_losses"]-offset,label="Test Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
        return
    return fig
