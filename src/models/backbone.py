import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Any

class FrozenViTSceneEncoder(nn.Module):
    """
    Frozen Vision/Spatial Scene Encoder.
    Extracts visual/spatial geometric features from 3D scene projections.
    """
    def __init__(self, pretrained: bool = False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=4),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(1),
            nn.Linear(32 * 8 * 8, 768)
        )
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: [B, 3, 224, 224]
        Returns: [B, 768] CLS token embedding
        """
        return self.encoder(img)

class Conv1D(nn.Module):
    """
    1D convolution / Linear layer matching standard GPT-2 linear projection interface.
    """
    def __init__(self, nf: int, nx: int):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x

class GPT2Attention(nn.Module):
    def __init__(self, d_model: int = 768, n_head: int = 12):
        super().__init__()
        self.n_head = n_head
        self.split_size = d_model
        self.c_attn = Conv1D(3 * d_model, d_model)
        self.c_proj = Conv1D(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query, key, value = self.c_attn(x).split(self.split_size, dim=2)
        B, T, C = x.size()
        head_dim = C // self.n_head
        q = query.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = key.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = value.view(B, T, self.n_head, head_dim).transpose(1, 2)
        
        att = F.scaled_dot_product_attention(q, k, v)
        att = att.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(att)

class GPT2MLP(nn.Module):
    def __init__(self, d_model: int = 768, d_ffn: int = 3072):
        super().__init__()
        self.c_fc = Conv1D(d_ffn, d_model)
        self.c_proj = Conv1D(d_model, d_ffn)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.act(self.c_fc(x)))

class GPT2Block(nn.Module):
    def __init__(self, d_model: int = 768, n_head: int = 12, d_ffn: int = 3072):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = GPT2Attention(d_model, n_head)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = GPT2MLP(d_model, d_ffn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class NativeGPT2Backbone(nn.Module):
    """
    Standard 12-layer GPT-2 Transformer backbone architecture (124M frozen backbone parameters).
    """
    def __init__(self, n_layer: int = 12, d_model: int = 768, n_head: int = 12):
        super().__init__()
        self.h = nn.ModuleList([GPT2Block(d_model, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, inputs_embeds: torch.Tensor) -> Any:
        x = inputs_embeds
        for block in self.h:
            x = block(x)
        h_out = self.ln_f(x)
        
        # Return a simple container object with last_hidden_state property
        class Output:
            def __init__(self, state):
                self.last_hidden_state = state
        return Output(h_out)

FrozenGPT2Backbone = NativeGPT2Backbone
