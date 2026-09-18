import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class AdaptiveRankGatingHead(nn.Module):
    """
    Gumbel-Softmax rank selection gate over candidate discrete ranks (e.g. {2, 4, 8, 16, 32}).
    Supports continuous temperature annealing during training and hard argmax during evaluation.
    """
    def __init__(self, input_dim: int = 384, ranks: List[int] = [2, 4, 8, 16, 32]):
        super().__init__()
        self.ranks = ranks
        self.num_ranks = len(ranks)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.num_ranks)
        )
        
    def forward(self, cond_vector: torch.Tensor, tau: float = 1.0, hard: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        cond_vector: [B, input_dim]
        Returns:
          w: [B, num_ranks] gating weights
          logits: [B, num_ranks]
        """
        logits = self.gate(cond_vector)
        if self.training:
            w = F.gumbel_softmax(logits, tau=tau, hard=hard)
        else:
            idx = torch.argmax(logits, dim=-1)
            w = F.one_hot(idx, num_classes=self.num_ranks).to(cond_vector.dtype)
        return w, logits

class SceneConditionedHypernetwork(nn.Module):
    """
    Hypernetwork mapping joint Scene + Task conditioning vector to LoRA adapter weights
    across all 12 transformer layers.
    """
    def __init__(
        self,
        ranks: List[int] = [2, 4, 8, 16, 32],
        num_layers: int = 12,
        cond_dim: int = 384,
        hidden_dim: int = 512,
        latent_dim: int = 256
    ):
        super().__init__()
        self.ranks = ranks
        self.max_rank = max(ranks)
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        
        # Layer embedding to parameterize distinct depth representations
        self.layer_embeddings = nn.Embedding(num_layers, 16)
        
        # Generator MLP
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim + 16, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        
        # Low-rank generator projection heads (to latent space)
        self.head_A_attn = nn.Linear(hidden_dim, 2 * self.max_rank * latent_dim) # Q, V
        self.head_B_attn = nn.Linear(hidden_dim, 2 * latent_dim * self.max_rank)
        
        self.head_A_ffn = nn.Linear(hidden_dim, self.max_rank * latent_dim)
        self.head_B_ffn = nn.Linear(hidden_dim, latent_dim * self.max_rank)
        
        # Shared trainable projection bases mapping latent_dim (256) to GPT-2 dimensions
        self.P_in = nn.Parameter(torch.randn(latent_dim, 768) * 0.01)
        self.P_attn = nn.Parameter(torch.randn(768, latent_dim) * 0.01)
        self.P_ffn = nn.Parameter(torch.randn(3072, latent_dim) * 0.01)
        
    def forward(self, cond_vector: torch.Tensor):
        """
        cond_vector: [B, cond_dim]
        Returns:
          all_lora_A_attn, all_lora_B_attn, all_lora_A_ffn, all_lora_B_ffn (lists of 12 layer tensors)
        """
        B_size = cond_vector.shape[0]
        device = cond_vector.device
        
        # Batched computation across all 12 layers simultaneously
        cond_exp = cond_vector.unsqueeze(1).expand(B_size, self.num_layers, -1) # [B, 12, cond_dim]
        layer_emb = self.layer_embeddings.weight.unsqueeze(0).expand(B_size, self.num_layers, -1) # [B, 12, 16]
        mlp_in = torch.cat([cond_exp, layer_emb], dim=-1) # [B, 12, cond_dim + 16]
        
        feat = self.mlp(mlp_in) # [B, 12, hidden_dim]
        
        # Attention LoRA: Q and V
        A_attn_lat = self.head_A_attn(feat).view(B_size, self.num_layers, 2, self.max_rank, self.latent_dim)
        B_attn_lat = self.head_B_attn(feat).view(B_size, self.num_layers, 2, self.latent_dim, self.max_rank)
        
        all_A_attn = torch.matmul(A_attn_lat, self.P_in)    # [B, 12, 2, max_rank, 768]
        all_B_attn = torch.matmul(self.P_attn, B_attn_lat)  # [B, 12, 2, 768, max_rank]
        
        # FFN LoRA: c_fc
        A_ffn_lat = self.head_A_ffn(feat).view(B_size, self.num_layers, self.max_rank, self.latent_dim)
        B_ffn_lat = self.head_B_ffn(feat).view(B_size, self.num_layers, self.latent_dim, self.max_rank)
        
        all_A_ffn = torch.matmul(A_ffn_lat, self.P_in)   # [B, 12, max_rank, 768]
        all_B_ffn = torch.matmul(self.P_ffn, B_ffn_lat) # [B, 12, 3072, max_rank]
        
        all_lora_A_attn = [all_A_attn[:, l] for l in range(self.num_layers)]
        all_lora_B_attn = [all_B_attn[:, l] for l in range(self.num_layers)]
        all_lora_A_ffn = [all_A_ffn[:, l] for l in range(self.num_layers)]
        all_lora_B_ffn = [all_B_ffn[:, l] for l in range(self.num_layers)]
        
        return all_lora_A_attn, all_lora_B_attn, all_lora_A_ffn, all_lora_B_ffn
