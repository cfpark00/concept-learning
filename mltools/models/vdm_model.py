import numpy as np
import torch
import torch.nn as nn
from torch import autograd, Tensor
from typing import Optional, Tuple
from torch.special import expm1
from tqdm import trange
from torch.distributions.normal import Normal
from lightning.pytorch import LightningModule

from mltools.models.model_tools import (
    kl_std_normal,
    FixedLinearSchedule,
    SigmoidSchedule,
    LearnedLinearSchedule,
    NNSchedule,
)


class VDM(nn.Module):
    def __init__(
        self,
        score_model: nn.Module,
        noise_schedule: str = "fixed_linear",
        gamma_min: float = -13.3,
        gamma_max: float = 5.0,
        antithetic_time_sampling: bool = True,
        data_noise: float = 1.0e-3,
        p_cfg=None,
        w_cfg=None
    ):
        """Variational diffusion model, continuous time implementation of arxiv:2107.00630.

        Args:
            score_model (nn.Module): model used to denoise
            noise_schedule (str, optional): whether fixed_linear or learned noise schedules.
            Defaults to "fixed_linear".
            gamma_min (float, optional): minimum gamma value. Defaults to -13.3.
            gamma_max (float, optional): maximum gamma value. Defaults to 5.0.
            antithetic_time_sampling (bool, optional): whether to do antithetic time sampling.
            Defaults to True.
            data_noise (float, optional): noise in data, used for reconstruction loss.
            Defaults to 1.0e-3.

        Raises:
            ValueError: when noise_schedule not in (fixed_linear, learned_linear)
        """
        super().__init__()
        self.score_model = score_model
        self.data_noise = data_noise
        assert noise_schedule in [
            "fixed_linear",
            "learned_linear",
            "learned_nn",
            "sigmoid",
        ], f"Unknown noise schedule {noise_schedule}"
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        if noise_schedule == "fixed_linear":
            self.gamma = FixedLinearSchedule(self.gamma_min, self.gamma_max)
        elif noise_schedule == "learned_linear":
            self.gamma = LearnedLinearSchedule(self.gamma_min, self.gamma_max)
        elif noise_schedule == "learned_nn":
            self.gamma = NNSchedule(self.gamma_min, self.gamma_max)
        elif noise_schedule == "sigmoid":
            self.gamma = SigmoidSchedule(self.gamma_min, self.gamma_max)
        else:
            raise ValueError(f"Unknown noise schedule {noise_schedule}")
        self.antithetic_time_sampling = antithetic_time_sampling
        self.p_cfg = p_cfg
        self.w_cfg = w_cfg

    def variance_preserving_map(
        self, x: Tensor, times: Tensor, noise: Optional[Tensor] = None
    ) -> Tensor:
        """Add noise to data sample, in a variance preserving way Eq. 10 in arxiv:2107.00630

        Args:
            x (Tensor): data sample
            times (Tensor): time steps
            noise (Tensor, optional): noise to add. Defaults to None.

        Returns:
            Tensor: Noisy sample
        """
        with torch.enable_grad():  # Need gradient to compute loss even when evaluating
            times = times.view((times.shape[0],) + (1,) * (x.ndim - 1))
            gamma_t = self.gamma(times)
        alpha = self.alpha(gamma_t)
        sigma = self.sigma(gamma_t)
        if noise is None:
            noise = torch.randn_like(x)
        return alpha * x + noise * sigma, gamma_t

    def sample_zt_given_zs(self, zs, s, t, pos_mean=False):
        gamma_t = self.gamma(t)
        gamma_s = self.gamma(s)
        alpha_t = self.alpha(gamma_t)
        alpha_s = self.alpha(gamma_s)
        sigma_t = self.sigma(gamma_t)
        sigma_s = self.sigma(gamma_s)
        alpha_ts = alpha_t / alpha_s
        sigma_ts_sq = sigma_t**2 - (alpha_ts**2) * (sigma_s**2)
        if pos_mean:
            return alpha_ts * zs
        return alpha_ts * zs + torch.sqrt(sigma_ts_sq) * torch.randn_like(zs)

    def sample_times(
        self,
        batch_size: int,
        device: str,
    ) -> Tensor:
        """Sample diffusion times for batch, used for monte carlo estimates

        Args:
            batch_size (int): size of batch

        Returns:
            Tensor: times
        """
        if self.antithetic_time_sampling:
            t0 = np.random.uniform(0, 1 / batch_size)
            times = torch.arange(t0, 1.0, 1.0 / batch_size, device=device)
        else:
            times = torch.rand(batch_size, device=device)
        return times

    def get_diffusion_loss(
        self,
        gamma_t: Tensor,
        times: Tensor,
        pred_noise: Tensor,
        noise: Tensor,
        bpd_factor: float,
    ) -> float:
        """get loss for diffusion process. Eq. 17 in arxiv:2107.00630

        Args:
            gamma_t (Tensor): gamma at time t
            times (Tensor): time steps
            pred_noise (Tensor): noise prediction
            noise (Tensor): noise added

        Returns:
            float: diffusion loss
        """
        gamma_grad = autograd.grad(  # gamma_grad shape: (B, )
            gamma_t,  # (B, )
            times,  # (B, )
            grad_outputs=torch.ones_like(gamma_t),
            create_graph=True,
            retain_graph=True,
        )[0]
        pred_loss = (
            ((pred_noise - noise) ** 2).flatten(start_dim=1).sum(axis=-1)
        )  # (B, )
        return bpd_factor * 0.5 * pred_loss * gamma_grad

    def get_latent_loss(
        self,
        x: Tensor,
        bpd_factor: float,
    ) -> float:
        """Latent loss to ensure the prior is truly Gaussian

        Args:
            x (Tensor): data sample

        Returns:
            float: latent loss
        """
        gamma_1 = self.gamma(torch.tensor([1.0], device=x.device))
        sigma_1_sq = torch.sigmoid(gamma_1)
        mean_sq = (1 - sigma_1_sq) * x**2
        return bpd_factor * kl_std_normal(mean_sq, sigma_1_sq).flatten(start_dim=1).sum(
            axis=-1
        )

    def get_reconstruction_loss(
        self,
        x: Tensor,
        bpd_factor: float,
    ):
        """Measure reconstruction error

        Args:
            x (Tensor): data sample

        Returns:
            float: reconstruction loss
        """
        noise_0 = torch.randn_like(x)
        times = torch.tensor([0.0], device=x.device)
        z_0, gamma_0 = self.variance_preserving_map(
            x,
            times=times,
            noise=noise_0,
        )
        # Generate a sample for z_0 -> closest to the data
        alpha_0 = torch.sqrt(torch.sigmoid(-gamma_0))
        z_0_rescaled = z_0 / alpha_0
        return -bpd_factor * Normal(loc=z_0_rescaled, scale=self.data_noise).log_prob(
            x
        ).flatten(start_dim=1).sum(axis=-1)

    def get_loss(
        self,
        x: Tensor,
        noise: Optional[Tensor] = None,
        reduction="mean",
        **kwargs
    ) -> float:
        """Get loss for diffusion model. Eq. 11 in arxiv:2107.00630

        Args:
            x (Tensor): data sample
            conditioning (Optional[Tensor], optional): conditioning. Defaults to None.
            noise (Optional[Tensor], optional): noise. Defaults to None.

        Returns:
            float: loss
        """
        if self.p_cfg is not None:
            assert "v_conditionings" in kwargs, "Need v_conditionings to mask out"
            batch_size=x.shape[0]
            mask=torch.rand(batch_size)<self.p_cfg
            #print("masking out",mask.float().mean(),"of conditioning" )
            v_conditionings=kwargs["v_conditionings"]
            for v in v_conditionings:
                v[mask,:]=0.
            kwargs["v_conditionings"]=v_conditionings
        bpd_factor = 1 / (np.prod(x.shape[1:]) * np.log(2))
        # Sample from q(x_t | x_0) with random t.
        times = self.sample_times(
            x.shape[0],
            device=x.device,
        ).requires_grad_(True)
        if noise is None:
            noise = torch.randn_like(x)
        x_t, gamma_t = self.variance_preserving_map(x=x, times=times, noise=noise)
        # Predict noise added
        pred_noise = self.score_model(
            x_t,
            t=(gamma_t.squeeze() - self.gamma_min) / (self.gamma_max - self.gamma_min),
            **kwargs
        )

        # *** Diffusion loss
        diffusion_loss = self.get_diffusion_loss(
            gamma_t=gamma_t,
            times=times,
            pred_noise=pred_noise,
            noise=noise,
            bpd_factor=bpd_factor,
        )

        # *** Latent loss: KL divergence from N(0, 1) to q(z_1 | x)
        latent_loss = self.get_latent_loss(
            x=x,
            bpd_factor=bpd_factor,
        )

        # *** Reconstruction loss:  - E_{q(z_0 | x)} [log p(x | z_0)].
        recons_loss = self.get_reconstruction_loss(
            x=x,
            bpd_factor=bpd_factor,
        )

        # *** Overall loss, Shape (B, ).
        loss = diffusion_loss + latent_loss + recons_loss
        if reduction == "mean":
            metrics = {
                "elbo": loss.mean(),
                "diffusion_loss": diffusion_loss.mean(),
                "latent_loss": latent_loss.mean(),
                "reconstruction_loss": recons_loss.mean(),
            }
            return loss.mean(), metrics
        return loss, {
            "elbo": loss,
            "diffusion_loss": diffusion_loss,
            "latent_loss": latent_loss,
            "reconstruction_loss": recons_loss,
        }

    def alpha(self, gamma_t: Tensor) -> Tensor:
        """Eq. 4 arxiv:2107.00630

        Args:
            gamma_t (Tensor): gamma evaluated at t

        Returns:
            Tensor: alpha
        """
        return torch.sqrt(torch.sigmoid(-gamma_t))

    def sigma(self, gamma_t):
        """Eq. 3 arxiv:2107.00630

        Args:
            gamma_t (Tensor): gamma evaluated at t

        Returns:
            Tensor: sigma
        """
        return torch.sqrt(torch.sigmoid(gamma_t))

    def sample_zs_given_zt(
        self,
        zt: Tensor,
        t: Tensor,
        s: Tensor,
        return_ddnm=False,
        **kwargs
    ) -> Tensor:
        """Sample p(z_s|z_t, x) used for standard ancestral sampling. Eq. 34 in arxiv:2107.00630

        Args:
            z (Tensor): latent variable at time t
            conditioning (Tensor): conditioning for samples
            t (Tensor): time t
            s (Tensor): time s

        Returns:
            zs, samples for time s
        """
        gamma_t = self.gamma(t)
        gamma_s = self.gamma(s)
        c = -expm1(gamma_s - gamma_t)
        alpha_t = self.alpha(gamma_t)
        alpha_s = self.alpha(gamma_s)
        sigma_t = self.sigma(gamma_t)
        sigma_s = self.sigma(gamma_s)
        if self.w_cfg is None:
            pred_noise = self.score_model(
                zt,
                t=(gamma_t - self.gamma_min) / (self.gamma_max - self.gamma_min),
                **kwargs
            )
        else:
            assert "v_conditionings" in kwargs, "Need v_conditionings to mask out"
            v_conditionings=kwargs["v_conditionings"]
            v_conditionings_orig=[v.clone() for v in v_conditionings]
            for v in v_conditionings:
                v[:]=0.
            kwargs["v_conditionings"]=v_conditionings
            pred_noise_uncond = self.score_model(
                zt,
                t=(gamma_t - self.gamma_min) / (self.gamma_max - self.gamma_min),
                **kwargs
            )
            kwargs["v_conditionings"]=v_conditionings_orig
            pred_noise_cond=self.score_model(
                zt,
                t=(gamma_t - self.gamma_min) / (self.gamma_max - self.gamma_min),
                **kwargs
            )
            pred_noise=pred_noise_uncond+self.w_cfg*(pred_noise_cond-pred_noise_uncond)
            #print("Used w_cfg",self.w_cfg)
        if not return_ddnm:
            mean = alpha_s / alpha_t * (zt - c * sigma_t * pred_noise)
            scale = sigma_s * torch.sqrt(c)
            return mean + scale * torch.randn_like(zt)
        else:
            gamma_0 = self.gamma(torch.tensor([0.0], device=zt.device))
            alpha_0 = self.alpha(gamma_0)
            sigma_0 = self.sigma(gamma_0)
            c0 = -expm1(gamma_0 - gamma_t)
            x_0t = alpha_0 / alpha_t * (zt - c0 * sigma_t * pred_noise)
            alpha_ts = alpha_t / alpha_s
            sigma_ts_sq = sigma_t**2 - (alpha_ts**2) * (sigma_s**2)
            w_z = alpha_ts * (sigma_s / sigma_t) ** 2
            w_x_0t = alpha_s * sigma_ts_sq / (sigma_t) ** 2
            scale = torch.sqrt(sigma_ts_sq * (sigma_s / sigma_t) ** 2)
            return w_z, w_x_0t, x_0t, scale

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        n_sampling_steps: int,
        device: str,
        z: Optional[Tensor] = None,
        return_all=False,
        verbose=False,
        **kwargs
    ) -> Tensor:
        """Generate new samples given some conditioning vector

        Args:
            conditioning (Tensor): conditioning
            batch_size (int): number of samples in batch
            n_sampling_steps (int): number of sampling steps
            device (str, optional): device to run model. Defaults to "cpu".
            z (Optional[Tensor], optional): initial latent variable. Defaults to None.

        Returns:
            Tensor: generated sample
        """
        if z is None:
            z = torch.randn(
                (batch_size, *self.score_model.shape),
                device=device,
            )
        steps = torch.linspace(
            1.0,
            0.0,
            n_sampling_steps + 1,
            device=device,
        )
        if return_all:
            zs = []
        for i in (
            trange(n_sampling_steps, desc="sampling")
            if verbose
            else range(n_sampling_steps)
        ):
            z = self.sample_zs_given_zt(
                zt=z,
                t=steps[i],
                s=steps[i + 1],
                **kwargs
            )
            if return_all:
                zs.append(z)
        if return_all:
            return torch.stack(zs, dim=0)
        return z


class LightVDM(LightningModule):
    def __init__(
        self,
        score_model: nn.Module,
        learning_rate: float = 3.0e-4,
        weight_decay: float = 1.0e-5,
        n_sampling_steps: int = 250,
        gamma_min: float = -13.3,
        gamma_max: float = 5.0,
        draw_figure=None,
        **kwargs,
    ):
        """Variational diffusion wrapper for lightning

        Args:
            score_model (nn.Module): model used to denoise
            learning_rate (float, optional): initial learning rate. Defaults to 1.0e-4.
            n_sampling_steps (int, optional): number of steps used to sample validations.
            Defaults to 250.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["score_model", "draw_figure"])
        self.model = VDM(
            score_model=score_model,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            **kwargs,
        )
        self.draw_figure = draw_figure

    def forward(
        self, x, **kwargs
    ) -> Tensor:
        """get loss for samples

        Args:
            x (Tensor): data samples
            conditioning (Tensor): conditioning
            conditioning_values (Tensor): conditioning_values

        Returns:
            Tensor: loss
        """
        return self.model.get_loss(
            x=x, **kwargs
        )

    def evaluate(self, batch: Tuple, stage: str = None) -> Tensor:
        """get loss function

        Args:
            batch (Tuple): batch of examples
            stage (str, optional): training stage. Defaults to None.

        Returns:
            Tensor: loss function
        """
        x,conditioning,conditioning_values=self.batch_unpack(batch)
        loss, metrics = self(
            x=x,
            s_conditioning=conditioning,
            v_conditionings=conditioning_values,
        )
        if self.logger is not None:
            self.logger.log_metrics(metrics)
        return loss

    def training_step(
        self,
        batch: Tuple,
        batch_idx: int,
    ) -> Tensor:
        """Training

        Args:
            batch (Tuple): batch of examples
            batch_idx (int): batch idx

        Returns:
            Tensor: loss
        """
        return self.evaluate(batch, "train")

    def draw_samples(
        self,
        batch_size: int,
        n_sampling_steps: int = 250,
        verbose=False,
        return_all=False,
        **kwargs
    ) -> Tensor:
        """draw samples from model

        Args:
            batch_size (int): number of samples in batch
            n_sampling_steps (int): number of sampling steps used to generate validation
            samples

        Returns:
            Tensor: generated samples
        """

        return self.model.sample(
            batch_size=batch_size,
            n_sampling_steps=n_sampling_steps,
            device=self.device,
            verbose=verbose,
            return_all=return_all,
            **kwargs
        )
    def batch_unpack(self,batch):
        x = batch["x"]
        conditioning = batch["conditioning"]
        conditioning_values = batch["conditioning_values"]
        return x,conditioning,conditioning_values

    def validation_step(self, batch: Tuple, batch_idx: int) -> Tensor:
        """validate model

        Args:
            batch (Tuple): batch of examples
            batch_idx (int): idx for batch

        Returns:
            Tensor: loss
        """
        # sample images during validation and upload to wandb
        x,conditioning,conditioning_values=self.batch_unpack(batch)
        loss = self.evaluate(batch, "val")
        self.log_dict({"val_loss": loss})

        if batch_idx == 0:
            samples = self.draw_samples(
                batch_size=len(x),
                n_sampling_steps=self.hparams.n_sampling_steps,
                s_conditioning=conditioning,
                v_conditionings=conditioning_values,
            )
            batch_plot={"x":x,"conditioning":conditioning,"conditioning_values":conditioning_values}
            if self.draw_figure is not None:
                fig = self.draw_figure(batch_plot, samples)

                if self.logger is not None:
                    self.logger.experiment.log_figure(figure=fig)
        return loss

    def test_step(self, batch, batch_idx):
        """test model

        Args:
            batch (Tuple): batch of examples
            batch_idx (int): idx for batch

        Returns:
            Tensor: loss
        """
        return self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

