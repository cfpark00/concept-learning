import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedEncoder(nn.Module):
    def __init__(self, net,random_masks_func):
        super().__init__()
        self.net = net
        self.random_masks_func=random_masks_func

    def forward(self, x):
        x = self.net(x)
        return x

    def get_masked_x(self,data,mask_channels=None,input_mask=False):
        x=data["x"]
        batch_size=x.shape[0]
        masks=self.random_masks_func(batch_size).to(x.device)
        if mask_channels is not None:
            masks1d=masks
            masks=masks[:,:,None]*mask_channels[None,None,:]
        x_masked = x.clone()
        x_masked[masks] = 0
        if input_mask:
            x_masked = torch.cat([x_masked, masks1d.to(dtype=x_masked.dtype)[:,:,None]], dim=-1)
        return x_masked

    def masked_pred(self, data,mask_channels=None,input_mask=False):
        x=data["x"]
        x_masked=self.get_masked_x(data,mask_channels,input_mask)
        data["x"]=x_masked
        x_pred = self(data)
        if input_mask:
            x_pred=x_pred[...,:-1]
        return x[masks], x_pred[masks]
    
    def get_loss(self,data,mask_channels=None,input_mask=False):
        if mask_channels is not None:
            mask_channels_=torch.zeros(data["x"].shape[-1],dtype=bool,device=data["x"].device)
            mask_channels_[mask_channels]=True
            mask_channels=mask_channels_
        x, x_pred = self.masked_pred(data,mask_channels,input_mask=input_mask)
        loss = F.mse_loss(x, x_pred)
        return loss