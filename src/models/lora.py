import torch
import torch.nn as nn
from typing import List, Optional

class DynamicLoRAWrapper(nn.Module):
    """
    LoRA Wrapper that applies dynamic adapter weights generated on-the-fly
    by the scene/task hypernetwork, supporting continuous or hard discrete rank selection.
    Optimized vectorized implementation for high-throughput CPU/CUDA inference and training.
    """
    def __init__(self, original_layer: nn.Module, lora_type: str, alpha: float = 8.0):
        super().__init__()
        self.original_layer = original_layer
        self.lora_type = lora_type # 'c_attn' or 'c_fc'
        self.alpha = float(alpha)
        
        # Context dynamic weights set during model forward pass
        self.current_lora_A: Optional[torch.Tensor] = None
        self.current_lora_B: Optional[torch.Tensor] = None
        self.current_w: Optional[torch.Tensor] = None
        self.current_ranks: Optional[List[int]] = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.original_layer(x)
        
        lora_A = self.current_lora_A
        lora_B = self.current_lora_B
        w = self.current_w
        ranks = self.current_ranks
        
        if lora_A is None or lora_B is None or w is None or ranks is None:
            return out
            
        B_size, L_size, d_in = x.shape
        alpha = self.alpha
        max_rank = max(ranks)
        
        # Vectorized rank weighting: compute per-rank dimension scale factor
        rank_weights = torch.zeros(B_size, max_rank, device=x.device, dtype=x.dtype)
        for i, r_val in enumerate(ranks):
            rank_weights[:, :r_val] += (alpha / r_val) * w[:, i:i+1]
            
        scale = rank_weights.unsqueeze(1) # [B, 1, max_rank]
        
        if self.lora_type == 'c_fc':
            # lora_A: [B, max_rank, 768], lora_B: [B, 3072, max_rank]
            xA = torch.bmm(x, lora_A.transpose(1, 2)) # [B, L, max_rank]
            xA_scaled = xA * scale
            delta = torch.bmm(xA_scaled, lora_B.transpose(1, 2)) # [B, L, 3072]
            out = out + delta
            
        elif self.lora_type == 'c_attn':
            # lora_A: [B, 2, max_rank, 768], lora_B: [B, 2, 768, max_rank]
            xA_Q = torch.bmm(x, lora_A[:, 0].transpose(1, 2)) # [B, L, max_rank]
            xA_Q_scaled = xA_Q * scale
            delta_Q = torch.bmm(xA_Q_scaled, lora_B[:, 0].transpose(1, 2)) # [B, L, 768]
            
            xA_V = torch.bmm(x, lora_A[:, 1].transpose(1, 2)) # [B, L, max_rank]
            xA_V_scaled = xA_V * scale
            delta_V = torch.bmm(xA_V_scaled, lora_B[:, 1].transpose(1, 2)) # [B, L, 768]
            
            # Non-mutating out-of-place assembly (dramatically accelerates backward autograd)
            out_Q = out[:, :, :768] + delta_Q
            out_K = out[:, :, 768:1536]
            out_V = out[:, :, 1536:] + delta_V
            out = torch.cat([out_Q, out_K, out_V], dim=-1)
            
        return out
