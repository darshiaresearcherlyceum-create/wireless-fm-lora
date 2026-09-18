import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from src.models.backbone import NativeGPT2Backbone
from src.models.wireless_fm import EnvironmentAwareWirelessFM


# =====================================================================
# B1: Independent Per-Task Models (No Foundation Backbone)
# =====================================================================
class MLPTaskModel(nn.Module):
    def __init__(self, output_dim: int, is_seq: bool = True):
        super().__init__()
        self.is_seq = is_seq
        self.net = nn.Sequential(
            nn.Linear(1040, 256),
            nn.GELU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_seq:
            return self.net(x)
        else:
            x_flat = x.mean(dim=1)
            return self.net(x_flat)


class IndependentPerTaskModels(nn.Module):
    """
    Baseline B1: Five independent MLP-based task models with no shared backbone.
    """
    def __init__(self):
        super().__init__()
        self.models = nn.ModuleList([
            MLPTaskModel(1024, is_seq=True),  # Task 0: CE
            MLPTaskModel(16, is_seq=True),    # Task 1: DET
            MLPTaskModel(1024, is_seq=True),  # Task 2: PREC
            MLPTaskModel(16, is_seq=True),    # Task 3: DEC
            MLPTaskModel(24, is_seq=False)    # Task 4: LOC
        ])

    def forward(self, inputs: torch.Tensor, task_id: torch.Tensor) -> Dict[str, torch.Tensor]:
        t_id = task_id[0].item()
        out = self.models[t_id](inputs)
        return {f"out_{t_id}": out}

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]


# =====================================================================
# B2: Full Fine-Tuning (Trainable GPT-2 Backbone, No LoRA)
# =====================================================================
class FullFineTuningModel(nn.Module):
    """
    Baseline B2: Fully unfrozen GPT-2 backbone, no LoRA adapters.
    """
    def __init__(self, model_name: str = "gpt2"):
        super().__init__()
        self.gpt2 = NativeGPT2Backbone(n_layer=12, d_model=768, n_head=12)
        for p in self.gpt2.parameters():
            p.requires_grad = True

        self.input_proj = nn.Linear(1040, 768)

        # Task Heads
        self.head_ce = nn.Linear(768, 1024)
        self.head_det = nn.Linear(768, 16)
        self.head_prec = nn.Linear(768, 1024)
        self.head_dec = nn.Linear(768, 16)

        # Task 4: User Localization (with Semantic Coupling)
        self.pi_proj = nn.Linear(16 * 64, 256)
        self.head_loc = nn.Sequential(
            nn.Linear(768 + 256, 512),
            nn.GELU(),
            nn.Linear(512, 24)
        )

    def forward(self, inputs: torch.Tensor, task_id: torch.Tensor) -> Dict[str, torch.Tensor]:
        B_size = inputs.shape[0]
        inputs_proj = self.input_proj(inputs)
        gpt2_out = self.gpt2(inputs_embeds=inputs_proj)
        h_shared = gpt2_out.last_hidden_state

        t_id = task_id[0].item()
        if t_id == 0:
            return {"out_0": self.head_ce(h_shared)}
        elif t_id == 1:
            return {"out_1": self.head_det(h_shared)}
        elif t_id == 2:
            return {"out_2": self.head_prec(h_shared)}
        elif t_id == 3:
            return {"out_3": self.head_dec(h_shared)}
        elif t_id == 4:
            h_mean = h_shared.mean(dim=1)
            out_dec = self.head_dec(h_shared)
            b_pred = torch.sigmoid(out_dec)
            b_pred_flat = b_pred.reshape(B_size, -1)
            b_sem = self.pi_proj(b_pred_flat)
            out_loc = self.head_loc(torch.cat([h_mean, b_sem], dim=-1))
            return {"out_4": out_loc}

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]


# =====================================================================
# B3: Static LoRA (Fixed Rank r=8 on Frozen GPT-2 Backbone)
# =====================================================================
class StaticLoRAWrapper(nn.Module):
    def __init__(self, original_layer: nn.Module, lora_type: str, rank: int = 8):
        super().__init__()
        self.original_layer = original_layer
        self.lora_type = lora_type
        self.rank = rank

        d_in = 768
        if lora_type == "c_fc":
            d_out = 3072
        elif lora_type == "c_attn":
            d_out = 2304
        else:
            raise ValueError(f"Unknown LoRA layer type: {lora_type}")

        if lora_type == "c_fc":
            self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
            self.lora_B = nn.Parameter(torch.randn(d_out, rank) * 0.01)
        elif lora_type == "c_attn":
            self.lora_A_Q = nn.Parameter(torch.randn(rank, d_in) * 0.01)
            self.lora_B_Q = nn.Parameter(torch.randn(d_in, rank) * 0.01)
            self.lora_A_V = nn.Parameter(torch.randn(rank, d_in) * 0.01)
            self.lora_B_V = nn.Parameter(torch.randn(d_in, rank) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.original_layer(x)
        alpha = 1.0

        if self.lora_type == "c_fc":
            delta_W = (alpha / self.rank) * torch.matmul(self.lora_B, self.lora_A)
            delta = torch.matmul(x, delta_W.t())
            out = out + delta
        elif self.lora_type == "c_attn":
            delta_W_Q = (alpha / self.rank) * torch.matmul(self.lora_B_Q, self.lora_A_Q)
            delta_W_V = (alpha / self.rank) * torch.matmul(self.lora_B_V, self.lora_A_V)

            delta_Q = torch.matmul(x, delta_W_Q.t())
            delta_V = torch.matmul(x, delta_W_V.t())

            out[:, :, :768] = out[:, :, :768] + delta_Q
            out[:, :, 1536:] = out[:, :, 1536:] + delta_V

        return out


class StaticLoRAModel(nn.Module):
    """
    Baseline B3: Fixed-Rank Static LoRA (rank=8) on Frozen GPT-2 Backbone.
    """
    def __init__(self, rank: int = 8, model_name: str = "gpt2"):
        super().__init__()
        self.gpt2 = NativeGPT2Backbone(n_layer=12, d_model=768, n_head=12)
        for p in self.gpt2.parameters():
            p.requires_grad = False

        for i in range(12):
            block = self.gpt2.h[i]
            block.attn.c_attn = StaticLoRAWrapper(block.attn.c_attn, "c_attn", rank=rank)
            block.mlp.c_fc = StaticLoRAWrapper(block.mlp.c_fc, "c_fc", rank=rank)

        self.input_proj = nn.Linear(1040, 768)
        from src.models.heads import WirelessTaskHeads
        self.task_heads = WirelessTaskHeads(hidden_dim=768, num_subcarriers=64, semantic_coupling=False)

    def forward(self, inputs: torch.Tensor, task_id: torch.Tensor, snr: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B_size = inputs.shape[0]
        inputs_proj = self.input_proj(inputs)
        gpt2_out = self.gpt2(inputs_embeds=inputs_proj)
        h_shared = gpt2_out.last_hidden_state

        t_id = task_id[0].item()
        head_outputs = self.task_heads(h_shared, inputs=inputs, snr=snr)
        return {f"out_{t_id}": head_outputs[f"out_{t_id}"]}

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        trainable_params = []
        for name, param in self.named_parameters():
            if "original_layer" not in name and "gpt2" not in name:
                if param.requires_grad:
                    trainable_params.append(param)
            elif "lora_A" in name or "lora_B" in name:
                if param.requires_grad:
                    trainable_params.append(param)
        return trainable_params


# =====================================================================
# B4: Multi-Task Learning Without Continual Learning
# =====================================================================
class MTLWithoutContinualLearning(EnvironmentAwareWirelessFM):
    """
    Baseline B4: Environment-Aware Dynamic Multi-Task Model trained across tasks
    without Continual Learning (EWC regularization disabled, lambda=0.0).
    """
    def __init__(self, ranks: Optional[List[int]] = None, **kwargs):
        if ranks is None:
            ranks = [2, 4, 8, 16, 32]
        super().__init__(ranks=ranks, **kwargs)

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        return super().get_trainable_parameters()