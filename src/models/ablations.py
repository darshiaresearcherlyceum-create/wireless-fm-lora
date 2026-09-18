import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional

from src.models.backbone import NativeGPT2Backbone, FrozenViTSceneEncoder
from src.models.lora import DynamicLoRAWrapper
from src.models.hypernetwork import SceneConditionedHypernetwork, AdaptiveRankGatingHead
from src.models.heads import WirelessTaskHeads
from src.models.baselines import StaticLoRAModel


class AblationA1_NoScene(nn.Module):
    """
    Ablation A1: No Scene Encoder.
    - Removes ViT scene encoder and scene projection.
    - Hypernetwork conditioning replaces scene embedding with zeros [B, 256].
    - Hypernetwork and adaptive rank gating run conditioned strictly on task embedding.
    """
    def __init__(
        self,
        ranks: List[int] = [2, 4, 8, 16, 32],
        feature_dim: int = 1040,
        semantic_coupling: bool = False,
        lora_alpha: float = 1.0
    ):
        super().__init__()
        self.ranks = ranks
        self.lora_alpha = float(lora_alpha)
        
        # 1. Spatial Context: ViT removed; only Task Embeddings remain
        self.task_embeddings = nn.Embedding(5, 128)
        
        # 2. Dynamic Hypernetwork & Adaptive Rank Selector (input dim: 256 zeros + 128 task = 384)
        self.hypernetwork = SceneConditionedHypernetwork(ranks=ranks, num_layers=12, cond_dim=384)
        self.rank_gate = AdaptiveRankGatingHead(input_dim=384, ranks=ranks)
        
        # 3. GPT-2 Backbone (Frozen) with Dynamic LoRA Wrappers
        self.gpt2 = NativeGPT2Backbone(n_layer=12, d_model=768, n_head=12)
        for p in self.gpt2.parameters():
            p.requires_grad = False
            
        self.wrappers: List[DynamicLoRAWrapper] = []
        for i in range(12):
            block = self.gpt2.h[i]
            block.attn.c_attn = DynamicLoRAWrapper(block.attn.c_attn, 'c_attn', alpha=self.lora_alpha)
            self.wrappers.append(block.attn.c_attn)
            
            block.mlp.c_fc = DynamicLoRAWrapper(block.mlp.c_fc, 'c_fc', alpha=self.lora_alpha)
            self.wrappers.append(block.mlp.c_fc)
            
        # 4. Wireless Signal Input Projection
        self.input_proj = nn.Linear(feature_dim, 768)
        
        # 5. Output Prediction Heads
        self.task_heads = WirelessTaskHeads(hidden_dim=768, num_subcarriers=64, semantic_coupling=semantic_coupling)

    def forward(
        self,
        inputs: torch.Tensor,
        scene: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
        tau: float = 1.0,
        hard: bool = False,
        z_s: Optional[torch.Tensor] = None,
        snr: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        B = inputs.shape[0]
        # Scene embedding is replaced with zeros
        z_s = torch.zeros(B, 256, device=inputs.device, dtype=inputs.dtype)
        z_t = self.task_embeddings(task_id)
        cond = torch.cat([z_s, z_t], dim=-1) # [B, 384]
        
        w, rank_logits = self.rank_gate(cond, tau=tau, hard=hard)
        A_attn, B_attn, A_ffn, B_ffn = self.hypernetwork(cond)
        
        w_idx = 0
        for l in range(12):
            attn_wrap = self.wrappers[w_idx]
            attn_wrap.current_lora_A = A_attn[l]
            attn_wrap.current_lora_B = B_attn[l]
            attn_wrap.current_w = w
            attn_wrap.current_ranks = self.ranks
            w_idx += 1
            
            ffn_wrap = self.wrappers[w_idx]
            ffn_wrap.current_lora_A = A_ffn[l]
            ffn_wrap.current_lora_B = B_ffn[l]
            ffn_wrap.current_w = w
            ffn_wrap.current_ranks = self.ranks
            w_idx += 1
            
        in_proj = self.input_proj(inputs)
        gpt2_out = self.gpt2(inputs_embeds=in_proj)
        h_shared = gpt2_out.last_hidden_state
        
        for wrap in self.wrappers:
            wrap.current_lora_A = None
            wrap.current_lora_B = None
            wrap.current_w = None
            wrap.current_ranks = None
            
        head_outputs = self.task_heads(h_shared, inputs=inputs, z_s=z_s, snr=snr)
        head_outputs['w'] = w
        head_outputs['rank_logits'] = rank_logits
        return head_outputs

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        return [p for n, p in self.named_parameters() if "gpt2" not in n and p.requires_grad]


class AblationA2_NoAdaptiveRank(nn.Module):
    """
    Ablation A2: No Adaptive Rank.
    - Keeps hypernetwork.
    - Forces rank = 8 for all tokens/layers.
    - Removes Gumbel rank gating head.
    """
    def __init__(
        self,
        ranks: List[int] = [2, 4, 8, 16, 32],
        feature_dim: int = 1040,
        semantic_coupling: bool = False,
        lora_alpha: float = 1.0
    ):
        super().__init__()
        self.ranks = ranks
        self.lora_alpha = float(lora_alpha)
        self.fixed_rank_idx = self.ranks.index(8)
        
        # 1. Spatial & Task Context Encoders
        self.scene_encoder = FrozenViTSceneEncoder()
        self.scene_proj = nn.Linear(768, 256)
        self.task_embeddings = nn.Embedding(5, 128)
        
        # 2. Dynamic Hypernetwork (no rank gating head)
        self.hypernetwork = SceneConditionedHypernetwork(ranks=ranks, num_layers=12, cond_dim=384)
        
        # 3. GPT-2 Backbone (Frozen) with Dynamic LoRA Wrappers
        self.gpt2 = NativeGPT2Backbone(n_layer=12, d_model=768, n_head=12)
        for p in self.gpt2.parameters():
            p.requires_grad = False
            
        self.wrappers: List[DynamicLoRAWrapper] = []
        for i in range(12):
            block = self.gpt2.h[i]
            block.attn.c_attn = DynamicLoRAWrapper(block.attn.c_attn, 'c_attn', alpha=self.lora_alpha)
            self.wrappers.append(block.attn.c_attn)
            
            block.mlp.c_fc = DynamicLoRAWrapper(block.mlp.c_fc, 'c_fc', alpha=self.lora_alpha)
            self.wrappers.append(block.mlp.c_fc)
            
        # 4. Wireless Signal Input Projection
        self.input_proj = nn.Linear(feature_dim, 768)
        
        # 5. Output Prediction Heads
        self.task_heads = WirelessTaskHeads(hidden_dim=768, num_subcarriers=64, semantic_coupling=semantic_coupling)

    def forward(
        self,
        inputs: torch.Tensor,
        scene: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
        tau: float = 1.0,
        hard: bool = False,
        z_s: Optional[torch.Tensor] = None,
        snr: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        B = inputs.shape[0]
        if z_s is None:
            cls_emb = self.scene_encoder(scene)
            z_s = self.scene_proj(cls_emb)
            
        z_t = self.task_embeddings(task_id)
        cond = torch.cat([z_s, z_t], dim=-1)
        
        # Fixed rank = 8 for all samples (one-hot vector at rank 8)
        w = torch.zeros(B, len(self.ranks), device=inputs.device, dtype=inputs.dtype)
        w[:, self.fixed_rank_idx] = 1.0
        rank_logits = torch.zeros_like(w)
        
        A_attn, B_attn, A_ffn, B_ffn = self.hypernetwork(cond)
        
        w_idx = 0
        for l in range(12):
            attn_wrap = self.wrappers[w_idx]
            attn_wrap.current_lora_A = A_attn[l]
            attn_wrap.current_lora_B = B_attn[l]
            attn_wrap.current_w = w
            attn_wrap.current_ranks = self.ranks
            w_idx += 1
            
            ffn_wrap = self.wrappers[w_idx]
            ffn_wrap.current_lora_A = A_ffn[l]
            ffn_wrap.current_lora_B = B_ffn[l]
            ffn_wrap.current_w = w
            ffn_wrap.current_ranks = self.ranks
            w_idx += 1
            
        in_proj = self.input_proj(inputs)
        gpt2_out = self.gpt2(inputs_embeds=in_proj)
        h_shared = gpt2_out.last_hidden_state
        
        for wrap in self.wrappers:
            wrap.current_lora_A = None
            wrap.current_lora_B = None
            wrap.current_w = None
            wrap.current_ranks = None
            
        head_outputs = self.task_heads(h_shared, inputs=inputs, z_s=z_s, snr=snr)
        head_outputs['w'] = w
        head_outputs['rank_logits'] = rank_logits
        return head_outputs

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        return [p for n, p in self.named_parameters() if "scene_encoder" not in n and "gpt2" not in n and p.requires_grad]


class AblationA3_NoHypernet(StaticLoRAModel):
    """
    Ablation A3: No Hypernetwork.
    - Same frozen GPT-2 backbone.
    - Same task heads (WirelessTaskHeads).
    - Ordinary Static LoRA (rank=8) inside the exact same training recipe.
    """
    def __init__(
        self,
        rank: int = 8,
        feature_dim: int = 1040,
        semantic_coupling: bool = False
    ):
        super().__init__(rank=rank)
