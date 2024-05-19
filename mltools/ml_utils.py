import torch
import tqdm
import numpy as np
import time

def to_np(ten):
    return ten.detach().cpu().numpy()

def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    np.random.seed(seed)

def train(model,dl_tr,batch_to_kwargs,n_steps,callback_steps,callbacks,device,**kwargs):
    clip_grad_norm=kwargs.get("clip_grad_norm",0.01)
    verbose=kwargs.get("verbose",1)
    online_losses=[]
    train_losses=[]
    val_losses=[]
    ckpts=[]
    step=0

    model.to(device)
    pbar=tqdm.tqdm(total=n_steps,desc="Training",disable=verbose==0)
    time_tr=0
    time_callbacks=0
    timer=time.perf_counter()
    while True:
        model.train()
        for batch in dl_tr:
            loss=model.get_loss(**batch_to_kwargs(batch))
            model.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            model.optimizer.step()

            online_losses.append(loss.item())
            step+=1
            pbar.update()

            if step in callback_steps:
                time_tr+=time.perf_counter()-timer
                timer=time.perf_counter()
                model.eval()
                for callback in callbacks:
                    callback(step=step,model=model)
                model.train()
                time_callbacks+=time.perf_counter()-timer
                timer=time.perf_counter()
            if step==n_steps:
                break
        if step==n_steps:
            break
    pbar.close()

    return_dict={
        "callback_steps":callback_steps,
        "online_losses":online_losses,
        "time_tr":time_tr,
        "time_callbacks":time_callbacks
    }
    return return_dict