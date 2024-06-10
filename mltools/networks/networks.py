import torch
from torch import nn
from torch.nn import functional as F
import warnings
import math
import inspect
from fancy_einsum import einsum

from mltools.networks.network_tools import zero_init, get_conv, get_timestep_embedding
from mltools.networks.blocks import (
    AttnBlock,
    ResNetBlock,
    ResNetDown,
    ResNetUp,
    TransformerBlock,
    LayerNorm,
)
from mltools.models.configs import GPTConfig


# CNNs
class CUNet(nn.Module):
    def __init__(
        self,
        shape=(1, 256, 256),
        out_channels=None,
        chs=[48, 96, 192, 384],
        s_conditioning_channels: int = 0,
        v_conditioning_dims: list = [],
        v_conditioning_type="common_zerolinear",
        v_embedding_dim: int = 64,
        v_augment=False,
        v_embed_no_s_gelu=False,
        t_conditioning=False,
        t_embedding_dim=64,
        init_scale: float = 0.02,
        num_res_blocks: int = 1,
        norm_groups: int = 8,
        mid_attn=True,
        n_attention_heads: int = 4,
        dropout_prob: float = 0.1,
        conv_padding_mode: str = "zeros",
        verbose: int = 0,
    ):
        super().__init__()
        self.shape = shape
        self.chs = chs
        self.dim = len(self.shape) - 1
        self.in_channels = self.shape[0]
        if out_channels is None:
            self.out_channels = self.in_channels
        else:
            self.out_channels = out_channels
        self.s_conditioning_channels = s_conditioning_channels
        self.v_conditioning_dims = v_conditioning_dims
        self.v_conditioning_type = v_conditioning_type
        self.common, self.cond_proj_type = v_conditioning_type.split("_")
        self.common = self.common == "common"
        self.v_embedding_dim = v_embedding_dim
        self.v_augment = v_augment
        if self.v_augment:
            # print("augmenting v_conditioning")
            assert self.common
        self.v_embed_no_s_gelu = v_embed_no_s_gelu

        self.t_conditioning = t_conditioning
        self.t_embedding_dim = t_embedding_dim
        self.norm_groups = norm_groups
        self.mid_attn = mid_attn
        if self.mid_attn and self.dim == 3:
            raise ValueError("3D attention very highly discouraged.")
        self.n_attention_heads = n_attention_heads
        self.dropout_prob = dropout_prob
        self.verbose = verbose

        conditioning_dims = []
        if self.t_conditioning:
            self.t_conditioning_dim = int(4 * self.t_embedding_dim)
            self.embed_t_conditioning = nn.Sequential(
                nn.Linear(self.t_embedding_dim, self.t_conditioning_dim),
                nn.GELU(),
                nn.Linear(self.t_conditioning_dim, self.t_conditioning_dim),
                nn.GELU(),
            )
            conditioning_dims.append(self.t_conditioning_dim)

        if len(self.v_conditioning_dims) > 0:
            self.embeds_v_conditionings = nn.ModuleList()
            for v_conditioning_dim in self.v_conditioning_dims:
                if self.common:
                    dim_mlp = (
                        2 * self.v_embedding_dim
                        if self.v_augment
                        else self.v_embedding_dim
                    )
                    self.embeds_v_conditionings.append(
                        nn.Sequential(
                            nn.Linear(v_conditioning_dim, dim_mlp),
                            nn.GELU(),
                            (
                                zero_init(nn.Linear(dim_mlp, dim_mlp))
                                if self.v_augment
                                else nn.Linear(dim_mlp, dim_mlp)
                            ),
                            nn.GELU() if not self.v_embed_no_s_gelu else nn.Identity(),
                        )
                    )
                    conditioning_dims.append(self.v_embedding_dim)
                else:
                    self.embeds_v_conditionings.append(nn.Identity())
                    conditioning_dims.append(v_conditioning_dim)
        if len(conditioning_dims) == 0:
            conditioning_dims = None
        self.conditioning_dims = conditioning_dims

        self.conv_kernel_size = 3
        self.norm_eps = 1e-6
        self.norm_affine = True
        self.act = "gelu"
        self.num_res_blocks = num_res_blocks
        assert self.conv_kernel_size % 2 == 1, "conv_kernel_size must be odd"
        norm_params = dict(
            num_groups=self.norm_groups, eps=self.norm_eps, affine=self.norm_affine
        )
        assert self.act in ["gelu", "relu", "silu"], "act must be gelu or relu or silu"

        def get_act():
            if self.act == "gelu":
                return nn.GELU()
            elif self.act == "relu":
                return nn.ReLU()
            elif self.act == "silu":
                return nn.SiLU()

        padding = self.conv_kernel_size // 2
        conv_params = dict(
            kernel_size=self.conv_kernel_size,
            padding=padding,
            padding_mode=conv_padding_mode,
        )
        nca_params = dict(
            norm_params=norm_params, get_act=get_act, conv_params=conv_params
        )
        resnet_params = dict(
            dim=self.dim,
            conditioning_dims=self.conditioning_dims,
            dropout_prob=self.dropout_prob,
            nca_params=nca_params,
            cond_proj_type=self.cond_proj_type,
        )

        self.n_sizes = len(self.chs)
        self.conv_in = get_conv(
            self.in_channels + self.s_conditioning_channels,
            self.chs[0],
            dim=self.dim,
            **conv_params,
        )

        # down
        self.downs = nn.ModuleList()
        for i_level in range(self.n_sizes):
            ch_in = chs[0] if i_level == 0 else chs[i_level - 1]
            ch_out = chs[i_level]
            resnets = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                resnets.append(ResNetBlock(ch_in, ch_out, **resnet_params))
                ch_in = ch_out
            down = ResNetDown(resnets)
            self.downs.append(down)

        # middle
        self.mid1 = ResNetBlock(ch_out, ch_out, **resnet_params)
        if self.mid_attn:
            self.mid_attn1 = AttnBlock(
                ch_out,
                n_heads=self.n_attention_heads,
                dim=self.dim,
                norm_params=norm_params,
            )
        # when no pad 1x262x262
        self.mid2 = ResNetBlock(ch_out, ch_out, **resnet_params)

        # upsampling
        self.ups = nn.ModuleList()
        ch_skip = 0
        for i_level in reversed(range(self.n_sizes)):
            ch_in = self.chs[i_level]
            ch_out = self.chs[0] if i_level == 0 else self.chs[i_level - 1]  # for up
            resnets = nn.ModuleList()
            for i_resnet in range(self.num_res_blocks):
                resnets.append(
                    ResNetBlock(
                        ch_in + (ch_skip if i_resnet == 0 else 0),
                        ch_in,
                        **resnet_params,
                    )
                )
            up = ResNetUp(resnet_blocks=resnets, ch_out=ch_out)
            ch_skip = ch_out
            self.ups.append(up)

        self.norm_out = nn.GroupNorm(num_channels=ch_out, **norm_params)
        self.act_out = get_act()
        self.conv_out = get_conv(
            in_channels=ch_out,
            out_channels=self.out_channels,
            dim=self.dim,
            init=zero_init,
            **conv_params,
        )

        if self.in_channels != self.out_channels:
            self.conv_residual_out = get_conv(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                dim=self.dim,
                init=zero_init,
                **conv_params,
            )

        for n, p in self.named_parameters():
            p.data *= init_scale

    def forward(
        self,
        x,
        t=None,
        s_conditioning=None,
        v_conditionings=None,
        alpha=False,
        beta=1,
        gamma=None,
    ):
        if s_conditioning is not None:
            if self.s_conditioning_channels != s_conditioning.shape[1]:
                raise ValueError(
                    f"Expected s_conditioning to have {self.s_conditioning_channels} channels, but got {s_conditioning.shape[1]}"
                )
            x_concat = torch.concat(
                (x, s_conditioning),
                axis=1,
            )
        else:
            x_concat = x

        conditionings = []
        if t is not None:
            if not self.t_conditioning:
                raise ValueError("t is not None, but t_conditioning is False")
            t = t.expand(
                x_concat.shape[0]
            ).clone()  # this clone has to be done for the t_embedding step
            assert t.shape == (x_concat.shape[0],)

            t_embedding = get_timestep_embedding(t, self.t_embedding_dim)

            t_cond = self.embed_t_conditioning(t_embedding)
            conditionings.append(t_cond)
        else:
            assert not self.t_conditioning, "t is None, but t_conditioning is True"

        if v_conditionings is not None:
            if len(v_conditionings) != len(self.v_conditioning_dims):
                raise ValueError(
                    f"Expected {len(self.v_conditioning_dims)} v_conditionings, but got {len(v_conditionings)}"
                )

            for i, v_conditioning in enumerate(v_conditionings):
                # v_conditioning: [64, 11]
                if v_conditioning.shape[1] != self.v_conditioning_dims[i]:
                    raise ValueError(
                        f"Expected v_conditioning to have {self.v_conditioning_dims[i]} channels, but got {v_conditioning.shape[1]}"
                    )

                v_cond = self.embeds_v_conditionings[i](v_conditioning)
                if self.v_augment:
                    means = v_cond[:, ::2]
                    stds = torch.exp(v_cond[:, 1::2])
                    v_cond = means + stds * torch.randn_like(stds)

                # v_cond: [64, 64]
                if gamma is not None:
                    blue_cond = torch.zeros((11))
                    blue_cond[4] = 0.1
                    blue_cond[5] = 0.1
                    blue_cond[6] = 0.9
                    blue_cond = blue_cond.unsqueeze(0)

                    blue_vec = self.embeds_v_conditionings[i](blue_cond.to(v_cond.device)).squeeze()
                    blue_vec = blue_vec / torch.sqrt((blue_vec**2).sum())

                    blue_scales = einsum(
                        "batch dim, dim -> batch", v_cond, blue_vec
                    )
                    blue_comps = einsum(
                        "batch, batch dim -> batch dim",
                        blue_scales, blue_vec.unsqueeze(0).repeat((v_cond.shape[0], 1))
                    )

                    v_cond = v_cond + (gamma * blue_comps)

                conditionings.append(v_cond)

        if len(conditionings) == 0:
            conditionings = None

        h = x_concat  # (B, C, H, W, D)

        h = self.conv_in(x_concat)
        # print(h.shape)
        skips = []
        for i, down in enumerate(self.downs):
            # conditionings[0]: t_emb
            # conditionings[1]: c_emb, [4, 11]
            h, h_skip = down(
                h,
                conditionings=conditionings,
                no_down=(i == (len(self.downs) - 1)),
                alpha=alpha,
                beta=beta,
            )
            # print(i,h.shape)
            if h_skip is not None:
                skips.append(h_skip)
        # print("total skips:",len(skips),[skip.shape for skip in skips])

        # middle
        h = self.mid1(h, conditionings=conditionings, alpha=alpha, beta=beta)
        # print("m1",h.shape)
        if self.mid_attn:
            h = self.mid_attn1(h)
            # print("ma1",h.shape)
        h = self.mid2(h, conditionings=conditionings, alpha=alpha, beta=beta)
        # print("m2",h.shape)

        # upsampling
        for i, up in enumerate(self.ups):
            x_skip = skips.pop() if len(skips) > 0 else None
            h = up(
                h,
                x_skip=x_skip,
                conditionings=conditionings,
                no_up=(i == self.n_sizes - 1),
                alpha=alpha,
                beta=beta,
            )
            # print(i,h.shape)
        h = self.norm_out(h)
        h = self.act_out(h)
        h = self.conv_out(h)
        if self.in_channels != self.out_channels:
            x = self.conv_residual_out(x)
        return h + x


# MLPs
class CMLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim=None,
        h_dims=[64],
        v_conditioning_dims: list = [],
        t_conditioning=False,
        t_embedding_dim=64,
        act="gelu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.h_dims = h_dims
        self.shape = (in_dim,)
        if out_dim is None:
            self.out_dim = self.in_dim
        else:
            self.out_dim = out_dim
        self.dims = [self.in_dim] + self.h_dims + [self.out_dim]
        self.v_conditioning_dims = v_conditioning_dims
        self.t_conditioning = t_conditioning
        self.t_embedding_dim = t_embedding_dim

        conditioning_dims = []
        if self.t_conditioning:
            self.t_conditioning_dim = int(4 * self.t_embedding_dim)
            self.embed_t_conditioning = nn.Sequential(
                nn.Linear(self.t_embedding_dim, self.t_conditioning_dim),
                nn.GELU(),
                nn.Linear(self.t_conditioning_dim, self.t_conditioning_dim),
            )
            conditioning_dims.append(self.t_conditioning_dim)
        if len(self.v_conditioning_dims) > 0:
            for v_conditioning_dim in self.v_conditioning_dims:
                conditioning_dims.append(v_conditioning_dim)
        self.conditioning_dims = conditioning_dims

        assert act in ["gelu", "relu", "silu"], "act must be gelu or relu or silu"
        if act == "gelu":
            self.act = nn.GELU()
        elif act == "relu":
            self.act = nn.ReLU()
        elif act == "silu":
            self.act = nn.SiLU()

        self.embedders = nn.ModuleList()
        self.layers = nn.ModuleList()
        for i, (dim_in, dim_out) in enumerate(zip(self.dims[:-1], self.dims[1:])):
            if i != len(self.dims) - 2:  # skip last layers
                embedders = nn.ModuleList()
                for conditioning_dim in self.conditioning_dims:
                    embedder = nn.Sequential(
                        nn.Linear(conditioning_dim, dim_out),
                        nn.GELU(),
                        nn.Linear(dim_out, dim_out),
                        nn.GELU(),
                    )
                    embedders.append(embedder)
                self.embedders.append(embedders)
            self.layers.append(nn.Linear(dim_in, dim_out))

    def forward(self, x, t=None, v_conditionings=None):

        conditionings = []
        if t is not None:
            if not self.t_conditioning:
                raise ValueError("t is not None, but t_conditioning is False")
            t = t.expand(
                x.shape[0]
            ).clone()  # this clone has to be done for the t_embedding step
            assert t.shape == (x.shape[0],)

            t_embedding = get_timestep_embedding(t, self.t_embedding_dim)

            t_cond = self.embed_t_conditioning(t_embedding)
            conditionings.append(t_cond)
        else:
            assert not self.t_conditioning, "t is None, but t_conditioning is True"
        """
        if v_conditionings is not None:
            if len(v_conditionings) != len(self.v_conditioning_dims):
                raise ValueError(
                    f"Expected {len(self.v_conditioning_dims)} v_conditionings, but got {len(v_conditionings)}"
                )
            for i, v_conditioning in enumerate(v_conditionings):
                if v_conditioning.shape[1] != self.v_conditioning_dims[i]:
                    raise ValueError(
                        f"Expected v_conditioning to have {self.v_conditioning_dims[i]} channels, but got {v_conditioning.shape[1]}"
                    )
                v_cond = self.embeds_v_conditionings[i](v_conditioning)
                conditionings.append(v_cond)
        """
        if v_conditionings is not None:
            for v_conditioning in v_conditionings:
                assert v_conditioning.shape[0] == x.shape[0], "batch not matching"
                conditionings.append(v_conditioning)
        assert len(conditionings) == len(
            self.embedders[0]
        ), "Number of conditionings must match number of embedders"

        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < (len(self.layers) - 1):
                embedders = self.embedders[i]
                for embedder, conditioning in zip(embedders, conditionings):
                    h = h + embedder(conditioning)
                h = self.act(h)
        return h + x


###Transformers
class Transformer(nn.Module):
    def __init__(self, config, embedder=None, unembedder=None):
        super().__init__()
        assert config.in_size is not None
        assert config.block_size is not None
        self.config = config

        if not self.config.pos_embed:
            print("Not using pos embed")

        if embedder is None:
            self.embedder = None
            self.transformer = nn.ModuleDict(
                dict(
                    wte=(
                        nn.Embedding(config.in_size, config.n_embd)
                        if config.tokenized
                        else nn.Linear(config.in_size, config.n_embd)
                    ),
                    wpe=(
                        nn.Embedding(config.block_size, config.n_embd)
                        if config.pos_embed
                        else None
                    ),
                    drop=nn.Dropout(config.dropout),
                    h=nn.ModuleList(
                        [TransformerBlock(config) for _ in range(config.n_layer)]
                    ),
                    ln_f=LayerNorm(config.n_embd, bias=config.bias),
                )
            )
            self.lm_head = nn.Linear(config.n_embd, config.in_size, bias=False)
            # with weight tying when using torch.compile() some warnings get generated:
            # "UserWarning: functional_call was passed multiple values for tied weights.
            # This behavior is deprecated and will be an error in future versions"
            # not 100% sure what this is, so far seems to be harmless. TODO investigate
            self.transformer.wte.weight = (
                self.lm_head.weight
            )  # https://paperswithcode.com/method/weight-tying
        else:
            assert unembedder is not None
            self.embedder = embedder
            for key, embedder in self.embedder.items():
                if isinstance(embedder, nn.Module):
                    self.add_module("embedder_" + key, embedder)
                    self.embedder[key] = getattr(self, "embedder_" + key)
            self.unembedder = unembedder
            for unembedder in self.unembedder.items():
                if isinstance(unembedder, nn.Module):
                    self.add_module("unembedder_" + key, unembedder)
                    self.unembedder[key] = getattr(self, "unembedder_" + key)
            self.transformer = nn.ModuleDict(
                dict(
                    wpe=(
                        nn.Embedding(config.block_size, config.n_embd)
                        if config.pos_embed
                        else None
                    ),
                    drop=nn.Dropout(config.dropout),
                    h=nn.ModuleList(
                        [TransformerBlock(config) for _ in range(config.n_layer)]
                    ),
                    ln_f=LayerNorm(config.n_embd, bias=config.bias),
                )
            )

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def forward(self, x):
        if self.embedder is not None:
            assert isinstance(x, dict)
            x_dict = x
            x = x_dict["x"]
        device = x.device
        if self.config.tokenized:
            b, t = x.size()
        else:
            b, t, c = x.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        if self.embedder is None:
            tok_emb = self.transformer.wte(
                x
            )  # token embeddings of shape (b, t, n_embd)
            if self.config.pos_embed:
                pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)
                pos_emb = self.transformer.wpe(
                    pos
                )  # position embeddings of shape (t, n_embd)
                tok_emb = tok_emb + pos_emb
            x = self.transformer.drop(tok_emb)
            for block in self.transformer.h:
                x = block(x)
            x = self.transformer.ln_f(x)
            logits = self.lm_head(x)
            return logits
        else:
            emb = self.embedder["x"](x)
            if self.config.pos_embed:
                pos = torch.arange(0, t, dtype=torch.long, device=device)
                if "pos" in self.embedder:
                    pos_emb = self.embedder["pos"](pos)
                else:
                    pos_emb = self.transformer.wpe(pos)
                emb = emb + pos_emb
            for key in x_dict.keys():
                if key in ["x", "pos"]:
                    continue
                emb = emb + self.embedder[key](x_dict[key])
            x = self.transformer.drop(emb)
            for block in self.transformer["h"]:
                x = block(x)
            x = self.transformer.ln_f(x)
            res = self.unembedder["x"](x)
            return res

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == "dropout" for k in override_args)
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing in_size=50257, block_size=1024, bias=True")
        config_args["in_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        config_args["bias"] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = Transformer(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )

            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


###VAEs
class Encoder(nn.Module):
    def __init__(
        self,
        shape,
        chs=[48, 96, 192],
        attn_sizes=[],
        mid_attn=False,
        num_res_blocks=1,
        dropout_prob=0.0,
        z_channels=4,
        double_z=True,
        n_attention_heads=1,
        norm_groups=8,
        norm_eps=1e-6,
        norm_affine=True,
        act="gelu",
        conv_kernel_size=3,
        conv_padding_mode="zeros",
    ):
        super().__init__()
        self.shape = shape
        self.in_channels = self.shape[0]
        assert self.shape[1] == self.shape[2], "input must be square"
        self.input_size = self.shape[1]
        self.chs = chs
        self.dim = len(self.shape) - 1
        self.attn_sizes = attn_sizes
        self.mid_attn = mid_attn
        if (len(self.attn_sizes) > 0 or self.mid_attn) and self.dim == 3:
            raise ValueError("3D attention very highly discouraged.")
        self.num_res_blocks = num_res_blocks
        self.dropout_prob = dropout_prob
        self.z_channels = z_channels
        self.double_z = double_z
        self.n_attention_heads = n_attention_heads

        assert conv_kernel_size % 2 == 1, "conv_kernel_size must be odd"
        norm_params = dict(num_groups=norm_groups, eps=norm_eps, affine=norm_affine)
        assert act in ["gelu", "relu", "silu"], "act must be gelu or relu or silu"

        def get_act():
            if act == "gelu":
                return nn.GELU()
            elif act == "relu":
                return nn.ReLU()
            elif act == "silu":
                return nn.SiLU()

        padding = conv_kernel_size // 2
        conv_params = dict(
            kernel_size=conv_kernel_size,
            padding=padding,
            padding_mode=conv_padding_mode,
        )
        nca_params = dict(
            norm_params=norm_params, get_act=get_act, conv_params=conv_params
        )
        resnet_params = dict(
            dim=self.dim,
            conditioning_dims=None,
            dropout_prob=self.dropout_prob,
            nca_params=nca_params,
        )

        self.n_sizes = len(self.chs)
        self.conv_in = get_conv(
            self.in_channels, self.chs[0], dim=self.dim, **conv_params
        )

        curr_size = self.input_size
        self.downs = nn.ModuleList()
        for i_level in range(self.n_sizes):
            ch_in = chs[0] if i_level == 0 else chs[i_level - 1]
            ch_out = chs[i_level]

            resnets = nn.ModuleList()
            attentions = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                resnets.append(ResNetBlock(ch_in, ch_out, **resnet_params))
                if curr_size in self.attn_sizes:
                    print("add attention")
                    attentions.append(
                        AttnBlock(
                            ch_out,
                            n_heads=self.n_attention_heads,
                            dim=self.dim,
                            norm_params=norm_params,
                        )
                    )
                ch_in = ch_out
            if len(attentions) == 0:
                attentions = None
            down = ResNetDown(resnets, attentions)
            curr_size = curr_size // 2
            self.downs.append(down)

        # middle

        # when no pad 1x266x266
        self.mid1 = ResNetBlock(ch_in, ch_in, **resnet_params)
        if self.mid_attn:
            self.mid_attn1 = AttnBlock(
                ch_in,
                n_heads=self.n_attention_heads,
                dim=self.dim,
                norm_params=norm_params,
            )
        # when no pad 1x262x262
        self.mid2 = ResNetBlock(ch_in, ch_in, **resnet_params)

        # end
        self.norm_out = nn.GroupNorm(num_channels=ch_in, **norm_params)
        self.act_out = get_act()
        # when no pad 1x258x258
        self.conv_out = get_conv(
            in_channels=ch_in,
            out_channels=2 * z_channels if double_z else z_channels,
            dim=self.dim,
            init=zero_init,
            **conv_params,
        )
        # when no pad 1x256x256

    def forward(self, x):
        # timestep embedding
        conditionings = None

        # downsampling
        h = self.conv_in(x)
        # print(h.shape)
        for i, down in enumerate(self.downs):
            h, _ = down(
                h, conditionings=conditionings, no_down=(i == (len(self.downs) - 1))
            )

        # middle
        h = self.mid1(h, conditionings=conditionings)
        if self.mid_attn:
            h = self.mid_attn1(h)
        h = self.mid2(h, conditionings=conditionings)

        # end
        h = self.norm_out(h)
        h = self.act_out(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        shape,
        chs=[48, 96, 192],
        attn_sizes=[],
        mid_attn=False,
        num_res_blocks=1,
        dropout_prob=0.0,
        z_channels=4,
        double_z=True,
        n_attention_heads=1,
        norm_groups=8,
        norm_eps=1e-6,
        norm_affine=True,
        act="gelu",
        conv_kernel_size=3,
        conv_padding_mode="zeros",
    ):
        super().__init__()
        self.shape = shape
        assert self.shape[1] == self.shape[2], "input must be square"
        self.in_channels = self.shape[0]
        self.input_size = self.shape[1]
        self.chs = chs
        self.dim = len(self.shape) - 1
        self.attn_sizes = attn_sizes
        self.mid_attn = mid_attn
        if (len(self.attn_sizes) > 0 or self.mid_attn) and self.dim == 3:
            raise ValueError("3D attention very highly discouraged.")
        self.num_res_blocks = num_res_blocks
        self.dropout_prob = dropout_prob
        self.z_channels = z_channels
        self.double_z = double_z
        self.n_attention_heads = n_attention_heads

        assert conv_kernel_size % 2 == 1, "conv_kernel_size must be odd"
        norm_params = dict(num_groups=norm_groups, eps=norm_eps, affine=norm_affine)
        assert act in ["gelu", "relu", "silu"], "act must be gelu or relu or silu"

        def get_act():
            if act == "gelu":
                return nn.GELU()
            elif act == "relu":
                return nn.ReLU()
            elif act == "silu":
                return nn.SiLU()

        padding = conv_kernel_size // 2
        conv_params = dict(
            kernel_size=conv_kernel_size,
            padding=padding,
            padding_mode=conv_padding_mode,
        )
        nca_params = dict(
            norm_params=norm_params, get_act=get_act, conv_params=conv_params
        )
        resnet_params = dict(
            dim=self.dim,
            conditioning_dims=None,
            dropout_prob=self.dropout_prob,
            nca_params=nca_params,
        )

        self.n_sizes = len(self.chs)

        ch_in = self.chs[-1]
        self.conv_in = get_conv(self.z_channels, ch_in, dim=self.dim, **conv_params)

        self.mid1 = ResNetBlock(ch_in, ch_in, **resnet_params)
        if self.mid_attn:
            self.mid_attn1 = AttnBlock(
                ch_in,
                n_heads=self.n_attention_heads,
                dim=self.dim,
                norm_params=norm_params,
            )
        self.mid2 = ResNetBlock(ch_in, ch_in, **resnet_params)

        # upsampling
        curr_size = self.input_size // 2 ** (self.n_sizes - 1)
        self.ups = nn.ModuleList()
        for i_level in reversed(range(self.n_sizes)):
            ch_in = self.chs[i_level]

            resnets = nn.ModuleList()
            attentions = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                resnets.append(ResNetBlock(ch_in, ch_in, **resnet_params))
                if curr_size in self.attn_sizes:
                    attentions.append(
                        AttnBlock(
                            ch_in,
                            n_heads=self.n_attention_heads,
                            dim=self.dim,
                            norm_params=norm_params,
                        )
                    )
            if len(attentions) == 0:
                attentions = None
            ch_out = self.chs[0] if i_level == 0 else self.chs[i_level - 1]  # for up
            up = ResNetUp(
                ch_out=ch_out, resnet_blocks=resnets, attention_blocks=attentions
            )
            curr_size = curr_size // 2
            self.ups.append(up)

        self.norm_out = nn.GroupNorm(num_channels=ch_out, **norm_params)
        self.act_out = get_act()
        self.conv_out = get_conv(
            in_channels=ch_out,
            out_channels=self.in_channels,
            dim=self.dim,
            init=zero_init,
            **conv_params,
        )

    def forward(self, z):
        self.last_z_shape = z.shape

        # timestep embedding
        conditionings = None

        # z to block_in
        h = self.conv_in(z)
        # print("after in",h.shape)

        # middle
        h = self.mid1(h, conditionings=conditionings)
        if self.mid_attn:
            h = self.mid_attn1(h)
        h = self.mid2(h, conditionings=conditionings)
        # print("after mid1,mid2",h.shape)

        # upsampling
        for i, up in enumerate(self.ups):
            h = up(h, conditionings=conditionings, no_up=(i == self.n_sizes - 1))
            # print(i,h.shape)

        h = self.norm_out(h)
        h = self.act_out(h)
        h = self.conv_out(h)
        return h
