# How to Reproduce Locked Paper Results

This guide provides end-to-end instructions for reproducing all official paper results using the clean repository structure.

---

## 1. Environment Setup

```bash
# Using conda
conda env create -f environment.yml
conda activate wireless-fm

# Or using pip
pip install -r requirements.txt
```

---

## 2. Checkpoint Storage Policy (Git / GitHub)

> **Important Note on Large Binary Weights:**  
> PyTorch checkpoint files (`.pt`) contain complete transformer and vision backbone weights (~375 MB to ~487 MB each) and exceed standard GitHub file size limits.  
> **Do not force-push or upload large checkpoints to GitHub.**  
> Keep only lightweight textual and tabular artifacts on GitHub:
> - `metrics.csv`
> - `run_log.txt`
> - `snr_evaluation.csv`
> - Localization range summaries (`.csv` and `.txt`)
> - Detection debug logs (`.csv` and `.txt`)

### Local Checkpoint Directory
Trained model checkpoints are stored locally at:
```text
results/checkpoints/
├── proposed_day 11_best.pt             (~487 MB)
├── static_lora_day11_best.pt           (~375 MB)
├── day13_A1_no_scene_best.pt           (~485 MB)
├── day13_A2_no_adaptive_rank_best.pt   (~487 MB)
└── day13_A3_no_hypernet_best.pt        (~375 MB)
```

---

## 3. Step-by-Step Reproduction Pipeline

### Step 1: Generate 3GPP Urban_Macro Dataset
Generates exactly 300 channel samples with the verified 210 train / 45 val / 45 test split:
```bash
python scripts/generate_data.py --samples 300 --env Urban_Macro --seed 42
```

### Step 2: Train Proposed Dynamic LoRA
Trains the frozen GPT-2 + scene-conditioned hypernetwork model on Urban_Macro for 10 epochs:
```bash
python scripts/train_multitask.py --env Urban_Macro --epochs 10 --config configs/proposed.yaml
```
- Outputs: [`logs/Day11/proposed/metrics.csv`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day11/proposed/metrics.csv), [`run_log.txt`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day11/proposed/run_log.txt), [`snr_evaluation.csv`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day11/proposed/snr_evaluation.csv).

### Step 3: Train Static LoRA Baseline (r=8)
Trains fixed-rank $r=8$ Static LoRA without hypernetwork:
```bash
python scripts/train_static_lora.py --env Urban_Macro --epochs 10 --config configs/static_lora.yaml
```
- Outputs: [`logs/Day11/static_lora/metrics.csv`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day11/static_lora/metrics.csv), [`run_log.txt`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day11/static_lora/run_log.txt), [`snr_evaluation.csv`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day11/static_lora/snr_evaluation.csv).

### Step 4: Evaluate Classical RZF Baseline
Evaluates closed-form Regularized Zero-Forcing (0 trainable parameters):
```bash
python scripts/run_rzf_baseline.py --env Urban_Macro --config configs/rzf_baseline.yaml
```
- Outputs: [`logs/Day11/rzf_baseline/snr_evaluation.csv`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day11/rzf_baseline/snr_evaluation.csv).

### Step 5: Execute Day 13 Ablations
Train each architectural ablation variant:
```bash
# A1: Without Scene Conditioning
python scripts/train_ablation.py --ablation A1_no_scene --epochs 10

# A2: Without Adaptive Rank (Forced r=8)
python scripts/train_ablation.py --ablation A2_no_adaptive_rank --epochs 10

# A3: Without Hypernetwork
python scripts/train_ablation.py --ablation A3_no_hypernet --epochs 10
```

---

## 4. Evaluation and Verification Tools

### Official SNR Grid Sweep
```bash
python scripts/eval_snr.py --ckpt results/checkpoints/proposed_day\ 11_best.pt --out logs/Day11/proposed/snr_evaluation.csv
```

### Localization Spatial Range & Center-Collapse Check
```bash
python scripts/eval_localization_range.py --ckpt results/checkpoints/proposed_day\ 11_best.pt
```

### MIMO Detection Bit-Level Debug (20 dB & 25 dB)
```bash
python scripts/eval_detection_debug.py --ckpt results/checkpoints/proposed_day\ 11_best.pt --snr 20
python scripts/eval_detection_debug.py --ckpt results/checkpoints/proposed_day\ 11_best.pt --snr 25
```
