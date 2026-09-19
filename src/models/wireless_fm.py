import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional

from src.models.backbone import FrozenViTSceneEncoder, NativeGPT2Backbone
from src.models.lora import DynamicLoRAWrapper
from src.models.hypernetwork import SceneConditionedHypernetwork, AdaptiveRankGatingHead
from src.models.heads import WirelessTaskHeads

class EnvironmentAwareWirelessFM(nn.Module):
    """
    Unified Environment-Aware Wireless Foundation Model.
    Integrates:
      - Frozen ViT-B/16 Scene Encoder
      - Learnable Task Embeddings
      - Dynamic LoRA Hypernetwork with Gumbel-Softmax Rank Selection (35.09M trainable Proposed parameters)
      - Frozen Pretrained GPT-2 Backbone (124M frozen backbone parameters) wrapped with Dynamic LoRA
      - 5 Task Heads with Semantic Coupling
    """
    def __init__(
        self,
        ranks: List[int] = [2, 4, 8, 16, 32],
        feature_dim: int = 1040,
        gpt2_model_name: str = "gpt2",
        semantic_coupling: bool = False,
        lora_alpha: float = 8.0
    ):
        super().__init__()
        self.ranks = ranks
        self.lora_alpha = float(lora_alpha)
        
        # 1. Spatial & Task Context Encoders
        self.scene_encoder = FrozenViTSceneEncoder()
        self.scene_proj = nn.Linear(768, 256)
        self.task_embeddings = nn.Embedding(5, 128)
        
        # 2. Dynamic Hypernetwork & Adaptive Rank Selector
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
        scene: torch.Tensor,
        task_id: torch.Tensor,
        tau: float = 1.0,
        hard: bool = False,
        z_s: Optional[torch.Tensor] = None,
        snr: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        inputs: [B, N_sub, feature_dim]
        scene: [B, 3, 224, 224]
        task_id: [B]
        """
        # 1. Conditioning Vector
        if z_s is None:
            cls_emb = self.scene_encoder(scene)
            z_s = self.scene_proj(cls_emb) # [B, 256]
            
        z_t = self.task_embeddings(task_id) # [B, 128]
        cond = torch.cat([z_s, z_t], dim=-1) # [B, 384]
        
        # 2. Dynamic Weight Generation & Rank Selection
        w, rank_logits = self.rank_gate(cond, tau=tau, hard=hard)
        A_attn, B_attn, A_ffn, B_ffn = self.hypernetwork(cond)
        
        # 3. Inject Weights into LoRA Wrappers
        w_idx = 0
        for l in range(12):
            # Attention wrapper
            attn_wrap = self.wrappers[w_idx]
            attn_wrap.current_lora_A = A_attn[l]
            attn_wrap.current_lora_B = B_attn[l]
            attn_wrap.current_w = w
            attn_wrap.current_ranks = self.ranks
            w_idx += 1
            
            # FFN wrapper
            ffn_wrap = self.wrappers[w_idx]
            ffn_wrap.current_lora_A = A_ffn[l]
            ffn_wrap.current_lora_B = B_ffn[l]
            ffn_wrap.current_w = w
            ffn_wrap.current_ranks = self.ranks
            w_idx += 1
            
        # 4. Forward through Backbone
        in_proj = self.input_proj(inputs) # [B, N_sub, 768]
        gpt2_out = self.gpt2(inputs_embeds=in_proj)
        h_shared = gpt2_out.last_hidden_state # [B, N_sub, 768]
        
        # Clean context hooks
        for wrap in self.wrappers:
            wrap.current_lora_A = None
            wrap.current_lora_B = None
            wrap.current_w = None
            wrap.current_ranks = None
            
        # 5. Output Heads
        head_outputs = self.task_heads(h_shared, inputs=inputs, z_s=z_s, snr=snr)
        head_outputs['w'] = w
        head_outputs['rank_logits'] = rank_logits
        return head_outputs

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """
        Returns strictly trainable parameters (hypernetwork, rank gate, task heads, projections).
        Excludes frozen ViT and GPT-2 weights.
        """
        return [p for n, p in self.named_parameters() if "scene_encoder" not in n and "gpt2" not in n and p.requires_grad]

    def get_hypernetwork_parameters(self) -> List[nn.Parameter]:
        """Alias for get_trainable_parameters() for backwards compatibility."""
        return self.get_trainable_parameters()

    def get_parameter_breakdown(self) -> Dict[str, Any]:
        """
        Returns exact parameter counts for all frozen backbones and trainable modules.
        """
        vit_params = sum(p.numel() for p in self.scene_encoder.parameters())
        gpt2_params = sum(p.numel() for p in self.gpt2.parameters())
        
        scene_proj_params = sum(p.numel() for p in self.scene_proj.parameters())
        task_emb_params = sum(p.numel() for p in self.task_embeddings.parameters())
        rank_gate_params = sum(p.numel() for p in self.rank_gate.parameters())
        hypernet_params = sum(p.numel() for p in self.hypernetwork.parameters())
        input_proj_params = sum(p.numel() for p in self.input_proj.parameters())
        task_heads_params = sum(p.numel() for p in self.task_heads.parameters())
        
        trainable_total = sum(p.numel() for p in self.get_trainable_parameters())
        frozen_total = vit_params + gpt2_params
        total_model_params = sum(p.numel() for p in self.parameters())
        
        return {
            'frozen_vit_scene_encoder': vit_params,
            'frozen_gpt2_backbone': gpt2_params,
            'trainable_scene_proj': scene_proj_params,
            'trainable_task_embeddings': task_emb_params,
            'trainable_rank_gate': rank_gate_params,
            'trainable_hypernetwork': hypernet_params,
            'trainable_input_proj': input_proj_params,
            'trainable_task_heads': task_heads_params,
            'total_trainable': trainable_total,
            'total_frozen': frozen_total,
            'total_parameters': total_model_params,
            'trainable_percentage': (trainable_total / total_model_params) * 100.0
        }
