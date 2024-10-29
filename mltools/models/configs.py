from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    tokenized: bool = True
    in_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True 
    causal: bool = True
    flash: bool = True
    verbose: int =1
    pos_embed: bool = True
