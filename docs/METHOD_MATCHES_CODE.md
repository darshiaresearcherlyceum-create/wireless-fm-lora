# Method-to-Code Alignment & Reviewer Verification Checklist

This document provides a reviewer-verifiable checklist mapping every core claim and method description in the paper directly to the concrete class, function, and file in the codebase.

---

## 1. Reviewer Verification Checklist

| Paper Sentence / Method Specification | Code File | Function / Class | Status |
| :--- | :--- | :--- | :---: |
| **Frozen GPT-2** | [`src/models/backbone.py`](file:///c:/Users/resea/Downloads/wireless-fm-lora/src/models/backbone.py)<br>[`src/models/wireless_fm.py`](file:///c:/Users/resea/Downloads/wireless-fm-lora/src/models/wireless_fm.py) | `NativeGPT2Backbone`<br>`EnvironmentAwareWirelessFM.__init__` (parameters set to `requires_grad=False`) | **Yes** |
| **Hypernetwork generates LoRA** | [`src/models/hypernetwork.py`](file:///c:/Users/resea/Downloads/wireless-fm-lora/src/models/hypernetwork.py) | `DynamicLoRAHypernetwork.forward` | **Yes** |
| **LoRA injected into Q, V, and first FFN** | [`src/models/lora.py`](file:///c:/Users/resea/Downloads/wireless-fm-lora/src/models/lora.py)<br>[`src/models/wireless_fm.py`](file:///c:/Users/resea/Downloads/wireless-fm-lora/src/models/wireless_fm.py) | `DynamicLoRAWrapper.forward` (adapts $Q$, $V$ in `c_attn`, and $c\_fc$ in `mlp`) | **Yes** |
| **Task ID embedding, not text prompt** | [`src/models/wireless_fm.py`](file:///c:/Users/resea/Downloads/wireless-fm-lora/src/models/wireless_fm.py) | `nn.Embedding(5, 128)` indexed by integer task ID (0 to 4) | **Yes** |
| **`semantic_coupling = false`** | [`configs/proposed.yaml`](file:///c:/Users/resea/Downloads/wireless-fm-lora/configs/proposed.yaml) | Line 47: `semantic_coupling: false` | **Yes** |
| **Detection uses MMSE init** | [`src/models/heads.py`](file:///c:/Users/resea/Downloads/wireless-fm-lora/src/models/heads.py) | `WirelessTaskHeads.forward` (Line 228: `out_det = base_syms + delta_det`, where `base_syms` is Linear MMSE) | **Yes** |
| **Precoding is neural, not RZF** | [`src/models/heads.py`](file:///c:/Users/resea/Downloads/wireless-fm-lora/src/models/heads.py) | `WirelessTaskHeads.forward` (Line 244: `W_pred_raw = self.head_prec(prec_in)`) | **Yes** |
| **EWC not used in official loss** | [`src/train/trainer.py`](file:///c:/Users/resea/Downloads/wireless-fm-lora/src/train/trainer.py)<br>[`scripts/train_multitask.py`](file:///c:/Users/resea/Downloads/wireless-fm-lora/scripts/train_multitask.py) | `train_one_epoch_multitask` (`ewc_lambda=0.0`) | **Yes** |
| **Rank penalty = 0** | [`configs/proposed.yaml`](file:///c:/Users/resea/Downloads/wireless-fm-lora/configs/proposed.yaml) | Line 60: `beta_rank_penalty: 0.0` | **Yes** |

---

## 2. Detailed Technical Audit of Each Row

### 1. Frozen GPT-2 Backbone
- **File**: [`src/models/backbone.py`](file:///c:/Users/resea/Downloads/wireless-fm-lora/src/models/backbone.py) and [`src/models/wireless_fm.py`](file:///c:/Users/resea/Downloads/wireless-fm-lora/src/models/wireless_fm.py)
- **Code verification**:
  ```python
  self.backbone = NativeGPT2Backbone(n_layer=12, d_model=768, n_head=12)
  for p in self.backbone.parameters():
      p.requires_grad = False
  ```
- **Audit**: All 85M parameters of the pre-trained transformer backbone are strictly frozen during multi-task adaptation.

### 2. Hypernetwork Generates LoRA
- **File**: [`src/models/hypernetwork.py`](file:///c:/Users/resea/Downloads/wireless-fm-lora/src/models/hypernetwork.py)
- **Class**: `DynamicLoRAHypernetwork`
- **Code verification**:
  Takes joint conditioning vector $\mathbf{z}_{\text{cond}} = [\mathbf{z}_{\text{scene}}, \mathbf{z}_{\text{task}}] \in \mathbb{R}^{384}$ and generates dynamic adapter tensors `A_attn, B_attn, A_ffn, B_ffn` on-the-fly for each layer.

### 3. LoRA Injection Targets ($Q, V$, and First FFN)
- **File**: [`src/models/lora.py`](file:///c:/Users/resea/Downloads/wireless-fm-lora/src/models/lora.py)
- **Class**: `DynamicLoRAWrapper`
- **Code verification**:
  ```python
  # Attention: Injects delta into Query (Q) and Value (V) only
  out_Q = out[:, :, :768] + delta_Q
  out_K = out[:, :, 768:1536]          # Key left frozen / unmodified
  out_V = out[:, :, 1536:] + delta_V
  out = torch.cat([out_Q, out_K, out_V], dim=-1)

  # MLP: Injects delta into first feedforward projection (c_fc, 768 -> 3072)
  delta = torch.bmm(xA_scaled, lora_B.transpose(1, 2))
  out = out + delta
  ```

### 4. Task ID Embedding (Not Text Prompts)
- **File**: [`src/models/wireless_fm.py`](file:///c:/Users/resea/Downloads/wireless-fm-lora/src/models/wireless_fm.py)
- **Code verification**:
  ```python
  self.task_embeddings = nn.Embedding(5, 128)
  z_t = self.task_embeddings(task_id)  # Task IDs 0 to 4 mapped directly to 128-dim vectors
  ```
- **Audit**: Does not use string prompts, tokenizers, or natural language prompts.

### 5. Semantic Coupling Setting
- **File**: [`configs/proposed.yaml`](file:///c:/Users/resea/Downloads/wireless-fm-lora/configs/proposed.yaml)
- **Code verification**: `semantic_coupling: false`
- **Audit**: Cross-task feature fusion between decoding and localization is turned off in the official proposed pipeline.

### 6. Detection Initialization (MMSE Init)
- **File**: [`src/models/heads.py`](file:///c:/Users/resea/Downloads/wireless-fm-lora/src/models/heads.py)
- **Code verification**:
  ```python
  # Linear MMSE equalized symbol calculation
  z_mmse = torch.linalg.solve(G_eff + 0.05 * eye_K, torch.matmul(H_eff.mH, Y_c)).squeeze(-1)
  base_syms = torch.cat([z_mmse_real, z_mmse_imag], dim=-1)
  out_det = base_syms + delta_det
  ```
- **Audit**: Zero error rate at high SNR ($\text{BER} = 0$ at 20–25 dB) is directly driven by the Linear MMSE base symbol initialization.

### 7. Neural Precoding (Not RZF)
- **File**: [`src/models/heads.py`](file:///c:/Users/resea/Downloads/wireless-fm-lora/src/models/heads.py)
- **Code verification**:
  ```python
  W_pred_raw = self.head_prec(prec_in)
  W_neural = torch.complex(w_real, w_imag)
  ```
- **Audit**: The proposed precoder outputs beamforming weights directly from a neural feedforward head without performing closed-form matrix inversion. Classical RZF is evaluated separately in [`scripts/run_rzf_baseline.py`](file:///c:/Users/resea/Downloads/wireless-fm-lora/scripts/run_rzf_baseline.py).

### 8. Official Loss (No EWC)
- **File**: [`src/train/trainer.py`](file:///c:/Users/resea/Downloads/wireless-fm-lora/src/train/trainer.py)
- **Code verification**: `ewc_lambda` is set to $0.0$ in official training invocations.
- **Audit**: No Elastic Weight Consolidation or continual-learning regularization is present in the official Urban_Macro multi-task loss.

### 9. Rank Complexity Penalty Setting
- **File**: [`configs/proposed.yaml`](file:///c:/Users/resea/Downloads/wireless-fm-lora/configs/proposed.yaml)
- **Code verification**: `beta_rank_penalty: 0.0`
- **Audit**: The complexity penalty is zero, allowing the model to allocate ranks without artificial regularizer pressure.
