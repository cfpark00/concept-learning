import torch
import torch.nn as nn
from lightning.pytorch import LightningModule

from mltools.networks.network_tools import get_conv
from mltools.networks.networks import Encoder, Decoder
from mltools.losses import MultiScaleMSE
from mltools.distributions import DiagonalGaussianDistribution


class AutoencoderKL(LightningModule):
    def __init__(
        self,
        enc_dec_params,
        embed_dim=8,
        learning_rate=1e-3,
        weight_decay=1.0e-5,
        nll_loss_type="l1",
        kl_weight=0.000001,
        draw_figure=None,
        **kwargs,
    ):
        super().__init__()
        # self.image_key = image_key
        self.save_hyperparameters(ignore=["draw_figure"])
        self.enc_dec_params = enc_dec_params
        self.encoder = Encoder(**self.enc_dec_params)
        self.decoder = Decoder(**self.enc_dec_params)
        self.dim = self.encoder.dim
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.nll_loss_type = nll_loss_type
        assert self.nll_loss_type in [
            "l1",
            "l2",
            "ms_mse",
        ], "nll_loss_type must be l1 or l2"
        if self.nll_loss_type == "ms_mse":
            self.ms_mse_exp = kwargs.get("ms_mse_exp", 1.0)
            self.ms_mse_loss = MultiScaleMSE(
                N=self.enc_dec_params["shape"][-1],
                dim=self.dim,
                k_func=lambda k: k**self.ms_mse_exp,
            )
        self.kl_weight = kl_weight
        self.draw_figure = draw_figure

        z_channels = self.encoder.z_channels
        self.quant_conv = get_conv(
            2 * z_channels, 2 * self.embed_dim, dim=self.dim, kernel_size=1, padding=0
        )
        self.post_quant_conv = get_conv(
            self.embed_dim, z_channels, dim=self.dim, kernel_size=1, padding=0
        )
        self.logvar = nn.Parameter(torch.zeros(size=(), dtype=torch.float32))

        ###kwargs
        self.kwargs=kwargs

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_loss(self, x):
        reconstructions, posteriors = self(x)
        if self.nll_loss_type == "l1":
            rec_loss = nn.functional.l1_loss(x, reconstructions, reduction="none")
        elif self.nll_loss_type == "l2":
            rec_loss = nn.functional.mse_loss(x, reconstructions, reduction="none")
        elif self.nll_loss_type == "ms_mse":
            rec_loss = self.ms_mse_loss(x, reconstructions)
        else:
            raise ValueError("nll_loss_type must be l1 or l2")
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        kl_loss = self.kl_weight * kl_loss
        loss = nll_loss + kl_loss
        metrics = {"kl_loss": kl_loss, "nll_loss": nll_loss, "loss": loss}
        return loss, metrics

    def evaluate(self, batch, stage):
        loss, metrics = self.get_loss(x=batch["x"])
        if self.logger is not None:
            self.logger.log_metrics(metrics)
        return loss

    def training_step(self, batch, batch_idx):
        return self.evaluate(batch, "train")

    def validation_step(self, batch, batch_idx):
        loss = self.evaluate(batch, "val")
        self.log_dict({"val_loss": loss.item()})
        # plot stuff
        if batch_idx == 0:
            reconstructions = self.get_reconstuctions(batch["x"])
            if self.draw_figure is not None:
                fig = self.draw_figure(batch, reconstructions)
                if self.logger is not None:
                    self.logger.experiment.log_figure(figure=fig)
        return loss

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, "test")

    @torch.no_grad()
    def get_reconstuctions(self, x, sample_posterior=False, return_posterior=False):
        reconstructions, posteriors = self(x, sample_posterior=sample_posterior)
        if return_posterior:
            return reconstructions, posteriors
        return reconstructions

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        if "reduce_lr_on_plateau" in self.kwargs:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.kwargs["reduce_lr_on_plateau"])
            lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
                "strict": True,
                }
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
        return optimizer
