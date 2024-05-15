import torch.nn as nn
import torch.nn.functional as F

class GPT(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer=transformer

        assert self.transformer.config.tokenized, "Model only supports tokenized input"

        if self.transformer.config.verbose>0:
            print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.transformer.parameters())
        if non_embedding:
            n_params -= self.transformer.transformer.wpe.weight.numel()
        return n_params

    def forward(self, x):
        return self.transformer(x)

    def get_loss(self, x, y):
        next_logits = self(x)
        loss = F.cross_entropy(next_logits.reshape(-1, next_logits.size(-1)), y.reshape(-1), ignore_index=-1)
        return loss
