import torch
import torch.nn as nn
import numpy as np

class GenDet(nn.Module):
    def __init__(self,net):
        super().__init__()
        self.net=net
        self.shape=net.shape

    def forward(self,c):
        z=torch.zeros(len(c),*self.shape).to(c.device)
        return self.net(x=z,v_conditionings=[c])
    
    def get_loss(self,c,x,reduction="mean"):
        x_rec=self(c)
        if reduction=="mean":
            loss=torch.nn.functional.mse_loss(x_rec,x)
            return loss
        elif reduction=="none":
            loss=torch.nn.functional.mse_loss(x_rec,x,reduction="none")
            return loss
    
    @torch.no_grad()
    def generate(self,c):
        return self(c)

class GenDiff(nn.Module):
    def __init__(self,net,beta_settings):
        super().__init__()
        self.net=net
        self.shape=net.shape
        self.schedule_noise(**beta_settings)

    def forward(self,x,c,t):
        return self.net(x=x,v_conditionings=[c],t=t)

    def get_loss(self,c,x):
        batch_size=x.shape[0]
        noise = torch.randn(x.shape).to(x.device)
        ts = torch.randint(0, self.T, (batch_size,)).long().to(x.device)
        noisy = self.add_noise(x, noise, ts)
        noise_pred = self(x=noisy, c=c, t=ts)
        loss=torch.nn.functional.mse_loss(noise_pred, noise)
        return loss
    
    @torch.no_grad()
    def generate(self,c):
        batch_size = c.shape[0]
        z = torch.randn(batch_size,*self.shape).to(c.device)
        timesteps = list(range(self.T))[::-1]
        for i, t in enumerate(timesteps):
            t=torch.tensor([t]*batch_size).long().to(c.device)
            noise_pred=self(x=z,c=c,t=t)
            z = self.step(noise_pred, t[0], z)
        return z
    
    @classmethod
    def get_schedule(cls,**kwargs):
        assert "type" in kwargs
        beta_schedule=kwargs["type"]
        assert "T" in kwargs
        T=kwargs["T"]

        if beta_schedule=="linear":
            assert "beta_i" in kwargs and "beta_f" in kwargs
            beta_i=kwargs.get("beta_i",1e-4)
            beta_f=kwargs.get("beta_f",0.02)
            betas=torch.linspace(beta_i,beta_f,T)
        elif beta_schedule=="quadratic":
            assert "beta_i" in kwargs and "beta_f" in kwargs
            beta_i=kwargs.get("beta_i",1e-4)
            beta_f=kwargs.get("beta_f",0.02)
            betas=torch.linspace(beta_i**0.5,beta_f**0.5,T)**2
        elif beta_schedule=="logsnr":
            assert "logsnr_i" in kwargs and "logsnr_f" in kwargs
            logsnr_i=kwargs["logsnr_i"]
            logsnr_f=kwargs["logsnr_f"]
            logsnrs=torch.linspace(logsnr_i,logsnr_f,T+1)
            alphabars=torch.sigmoid(logsnrs)
            betas=1-(alphabars[1:]/alphabars[:-1])
        elif beta_schedule=="gamma_linear":
            assert "gamma_min" in kwargs and "gamma_max" in kwargs
            gamma_min=kwargs["gamma_min"]
            gamma_max=kwargs["gamma_max"]
            gammas=torch.linspace(gamma_min,gamma_max,T+1)
            alphas_vdm=torch.sqrt(torch.sigmoid(-gammas))
            alphas_vdm_sq=alphas_vdm**2
            alphas=torch.cat([alphas_vdm_sq[0][None],alphas_vdm_sq[1:]/alphas_vdm_sq[:-1]],dim=0)
            betas=1-alphas


        alphas=1.0-betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        sqrt_alphas_cumprod = alphas_cumprod ** 0.5
        sqrt_1_m_alphas_cumprod = (1 - alphas_cumprod) ** 0.5
        sqrt_inv_alphas_cumprod = torch.sqrt(1/alphas_cumprod)
        sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(1/alphas_cumprod-1)
        posterior_mean_coef1=betas*torch.sqrt(alphas_cumprod_prev)/(1-alphas_cumprod)
        posterior_mean_coef2=(1-alphas_cumprod_prev)*torch.sqrt(alphas)/(1-alphas_cumprod)
        
        schedule={
            "betas":betas,
            "alphas":alphas,
            "alphas_cumprod":alphas_cumprod,
            "alphas_cumprod_prev":alphas_cumprod_prev,
            "sqrt_alphas_cumprod":sqrt_alphas_cumprod,
            "sqrt_1_m_alphas_cumprod":sqrt_1_m_alphas_cumprod,
            "sqrt_inv_alphas_cumprod":sqrt_inv_alphas_cumprod,
            "sqrt_inv_alphas_cumprod_minus_one":sqrt_inv_alphas_cumprod_minus_one,
            "posterior_mean_coef1":posterior_mean_coef1,
            "posterior_mean_coef2":posterior_mean_coef2
        }
        return schedule
    
    def schedule_noise(self,**kwargs):
        assert "T" in kwargs
        self.T=kwargs["T"]

        schedule=self.__class__.get_schedule(**kwargs)

        for key,item in schedule.items():
            self.register_buffer(key,item)

    def reconstruct_x0(self,x_t,t,noise):
        s1=self.sqrt_inv_alphas_cumprod[t]
        s2=self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1=s1.reshape(-1,1)
        s2=s2.reshape(-1,1)
        return s1*x_t-s2*noise

    def q_posterior(self,x_0,x_t,t):
        s1=self.posterior_mean_coef1[t]
        s2=self.posterior_mean_coef2[t]
        s1=s1.reshape(-1,1)
        s2=s2.reshape(-1,1)
        mu=s1*x_0+s2*x_t
        return mu
    
    def get_variance(self,t):
        if t==0:
            return 0
        variance=self.betas[t]*(1.-self.alphas_cumprod_prev[t])/(1.-self.alphas_cumprod[t])
        variance=variance.clip(1e-20)
        return variance
    
    def step(self,model_output,t,sample):
        pred_original_sample=self.reconstruct_x0(sample,t,model_output)#remove the noise,
        pred_prev_sample=self.q_posterior(pred_original_sample,sample,t)#interpolate to estimate previous

        variance=0
        if t>0:
            noise=torch.randn_like(model_output)
            variance=(self.get_variance(t)**0.5)*noise

        pred_prev_sample=pred_prev_sample+variance

        return pred_prev_sample
    
    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_1_m_alphas_cumprod[timesteps]

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)

        return s1 * x_start + s2 * x_noise

from mltools.models import vdm_model

class GenVDiff(nn.Module):
    def __init__(self,net,beta_settings,data_noise,p_cfg=None,w_cfg=None):
        super().__init__()
        self.shape=net.shape
        assert beta_settings["type"]=="logsnr"
        assert beta_settings["noise_schedule"] in ["fixed_linear","learned_linear","learned_nn"]
        assert "gamma_min" in beta_settings and "gamma_min" in beta_settings
        self.data_noise=data_noise
        self.model=vdm_model.VDM(score_model=net,
                                 noise_schedule=beta_settings["noise_schedule"],
                                 gamma_min=beta_settings["gamma_min"],
                                 gamma_max=beta_settings["gamma_max"],
                                 data_noise=self.data_noise,
                                 p_cfg=p_cfg,
                                 w_cfg=w_cfg)
    
    def get_loss(self,c,x,reduction="mean"):
        if reduction=="mean":
            return self.model.get_loss(x=x,v_conditionings=[c],reduction=reduction)[0]#1 is decomposed loss
        else:
            return self.model.get_loss(x=x,v_conditionings=[c],reduction=reduction)
    
    @torch.no_grad()
    def generate(self,c,T=50):
        batch_size=c.shape[0]
        return self.model.sample(batch_size=batch_size,device=c.device,n_sampling_steps=T,v_conditionings=[c])

    @classmethod
    def get_min_vlb_(cls,data_noise,n_dim):
        #bpd=1/(n_dim*np.log(2))
        #min_vlb=-np.log((2*np.pi*(data_noise**2))**(-n_dim/2))*(bpd)
        min_vlb=np.log(2*np.pi*(data_noise**2))/(2*np.log(2))
        return min_vlb
    
    def get_min_vlb(self):
        return self.__class__.get_min_vlb_(self.data_noise,np.prod(self.shape))

class Classifier(nn.Module):
    def __init__(self,net,n_classes=None,out_dim=None):
        super().__init__()
        self.net=net
        self.shape=net.shape
        if out_dim is not None:
            assert n_classes is None
            self.n_classes=None
            self.out_dim=out_dim
        else:
            self.n_classes=n_classes
            self.out_dim=np.sum(n_classes)
        self.out_channels=self.net.out_channels
        self.head=nn.Sequential(
            nn.GroupNorm(num_groups=4,num_channels=self.out_channels),
            nn.Linear(self.out_channels,64),
            nn.GELU(),
            nn.GroupNorm(num_groups=4,num_channels=64),
            nn.Linear(64,self.out_dim)
        )

    def unravel_index(self,index):
        out = []
        for dim in reversed(self.n_classes):
            out.append(index % dim)
            index = index // dim
        return tuple(reversed(out))
    
    def forward(self,x):
        x=self.net(x)
        x=torch.nn.functional.max_pool2d(x, kernel_size=x.size()[2:])[:,:,0,0]
        return self.head(x)

    def get_loss(self,x,l):
        logits=self(x)
        if self.n_classes is None:
            return nn.functional.mse_loss(logits,l)
        else:
            classes=self.unravel_index(l)
            loss=0
            start=0
            for i,n_class in enumerate(self.n_classes):
                loss+=nn.functional.cross_entropy(logits[:,start:start+n_class],classes[i])
                start+=n_class
        return loss
    
    def classify(self,x,return_probs=False,return_logits=False):
        assert self.n_classes is not None, "This is a regression model, use regress instead"
        if return_probs and return_logits:
            raise ValueError("Only one of return_probs and return_logits can be True")
        logits=self(x)
        classes=[]
        start=0
        for i,n_class in enumerate(self.n_classes):
            if return_probs:
                classes.append(nn.functional.softmax(logits[:,start:start+n_class],dim=1))
            elif return_logits:
                classes.append(logits[:,start:start+n_class])
            else:
                classes.append(torch.argmax(logits[:,start:start+n_class],dim=1))
            start+=n_class
        return classes
    
    def regress(self,x):
        return self(x)
        
