import numpy as np
import torch
import torch.nn as nn
from torch import autograd, Tensor
from typing import Optional, Tuple
from torch.special import expm1
import tqdm
from torch.distributions.normal import Normal
from lightning.pytorch import LightningModule

from mltools.models.model_tools import (
    kl_std_normal,
    FixedLinearSchedule,
    SigmoidSchedule,
    LearnedLinearSchedule,
    NNSchedule,
)

#assert False, "This code is not prepared yet."

class EulerSDE(nn.Module):
    def __init__(self, drift_func, sigma, delta_t):
        super().__init__()
        self.drift_func = drift_func
        self.sigma = sigma
        self.delta_t =delta_t

    def step_forward(self, x, x0, t: torch.tensor, is_last=False):
        """Euler-Maruyama."""
        if is_last:
            dW = 0.
        else:
            dW = torch.sqrt(self.delta_t)*torch.randn(size=x.shape).to(x.device)
        bt_term = self.drift_func(t, x, x0)*self.delta_t 
        sigma_term = self.sigma(t)*dW
        x += bt_term + sigma_term
        return x

    def integrate(self, x0,verbose=0):
        n_steps = int(1/self.delta_t)
        x = x0.clone()
        for i in tqdm.tqdm(range(n_steps),desc="Integrating...",disable=verbose==0):
            x = self.step_forward(x=x, x0=x0, t=i*self.delta_t, is_last=(i==n_steps-1))
        return x


class LeimkuhlerMatthewsSDE(nn.Module):
    def __init__(self, drift_func, sigma, delta_t):
        super().__init__()
        self.drift_func = drift_func
        self.sigma = sigma
        self.delta_t =delta_t

    def step_forward(self, x, x0, t: torch.tensor, is_last=False):
        """Leimkuhler-Matthews."""
        if is_last:
            dW = 0.
        else:
            dW = torch.sqrt(self.delta_t)*torch.randn(size=x.shape).to(x.device)/np.sqrt(2)
        bt_term = self.drift_func(t, x, x0)*self.delta_t 
        sigma_term = self.sigma(t)*dW
        x += bt_term + sigma_term
        return x

    def integrate(self, x0,verbose=0):
        n_steps = int(1/self.delta_t)
        x = x0.clone()
        for i in tqdm.tqdm(range(n_steps),desc="Integrating...",disable=verbose==0):
            x = self.step_forward(x=x, x0=x0, t=i*self.delta_t, is_last=(i==n_steps-1))
        return x
    

class SFM(torch.nn.Module):
    """
    Stochastic Flow Matching.
    """
    def __init__(
        self,
        velocity_model,
        noise_schedule: str = "default",
    ):
        super().__init__()
        self.velocity_model = velocity_model
        self.noise_schedule = noise_schedule

    def get_alpha_t(self, t):
        return 1 - t

    def get_beta_t(self, t):
        return t**2

    def get_sigma_t(self, t):
        return 1 - t

    def get_alpha_t_dot(self, t):
        #this must be the derivative of alpha_t
        return -torch.ones_like(t)

    def get_beta_t_dot(self, t):
        #this must be the derivative of beta_t
        return 2.0 * t

    def get_sigma_t_dot(self, t):
        #this must be the derivative of sigma_t
        return -torch.ones_like(t)

    def get_xt(self, x0, x1, t, epsilon):
        t = t.view(t.shape[0], *([1] * (x0.dim() - 1)))
        return (
            self.get_alpha_t(t) * x0
            + self.get_beta_t(t) * x1
            + torch.sqrt(t) * self.get_sigma_t(t) * epsilon
        )

    def get_rt(self, x0, x1, t, epsilon):
        t = t.view(t.shape[0], *([1] * (x0.dim() - 1)))
        return (
            self.get_alpha_t_dot(t) * x0
            + self.get_beta_t_dot(t) * x1
            + self.get_sigma_t_dot(t) * torch.sqrt(t) * epsilon
        )

    def compute_loss(
        self,
        x0,
        x1,
        h=None,
        t=None,
    ):
        if t is None:
            #randomly samples times from [0, 1)
            t = torch.rand(x0.shape[0]).type_as(x0)
        eps = torch.randn_like(x0)
        xt = self.get_xt(x0=x0, x1=x1, t=t, epsilon=eps)
        rt = self.get_rt(x0=x0, x1=x1, t=t, epsilon=eps)
        b_pred = self.velocity_model(xt, t=t, s_conditioning=x0, v_conditionings=h,)
        return torch.mean((b_pred - rt) ** 2)

    def predict(
        self,
        x0,
        h=None,
        n_sampling_steps=100,
        verbose=0
    ):
        delta_t = 1./n_sampling_steps
        x0_size = x0.size()
        def drift_func(t, xt, x0):
            return self.velocity_model(xt.view(x0_size), t=t,
                                       s_conditioning=x0.view(x0_size), v_conditionings=h,).flatten(
                start_dim=1
            )
        def diffusion_func(t,):
            return self.get_sigma_t(t)

        sde = EulerSDE(
            drift_func = drift_func,
            sigma=diffusion_func,
            delta_t = torch.tensor(delta_t).to(x0.device),
        )
        with torch.no_grad():
            ys = sde.integrate(x0.flatten(start_dim=1,),verbose=verbose)
            ys = ys.view(x0_size)
        return ys
    


class LightSFM(LightningModule):
    def __init__(
        self,
        velocity_model,
        learning_rate=1.0e-4,
        n_sampling_steps=100,
        draw_figure=None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["velocity_model", "draw_figure"])
        self.sfm = SFM(
            velocity_model=velocity_model,
        )
        self.draw_figure = draw_figure

    def get_loss(self, batch):
        x0=batch["x0"]
        x1=batch["x1"]
        conditioning_values=batch["conditioning_values"]
        loss=self.sfm.compute_loss(x0=x0, x1=x1, h=conditioning_values)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch=batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch=batch)
        if batch_idx == 0:
            x0=batch["x0"]
            x1=batch["x1"]
            conditioning_values=batch["conditioning_values"]

            samples = self.sfm.predict(x0=x0, h=conditioning_values,
                                       n_sampling_steps=self.hparams.n_sampling_steps)
            if self.draw_figure is not None:
                fig = self.draw_figure(batch, samples)

                if self.logger is not None:
                    self.logger.experiment.log_figure(figure=fig)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(
        self,
    ):
        optimizer = torch.optim.AdamW(
            self.sfm.parameters(), lr=self.hparams.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=5,
        )
        return optimizer

