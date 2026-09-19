# Environment-Aware Hypernetwork with a Frozen GPT-2 Backbone for Wireless Channel Estimation

## What this repo is

This repository is the official implementation of a frozen GPT-2 transformer backbone (124M frozen parameters) paired with an environment-aware, scene-conditioned hypernetwork that generates Low-Rank Adaptation (LoRA) weights (35.09M trainable Proposed parameters) for physical-layer wireless communications. All models and baselines are evaluated on the 3GPP TR 38.901 Urban Macro (`Urban_Macro`) propagation environment across a standard SNR grid from $-5$ to $25$ dB.

## What this repo is not

- not a 5-task foundation model that solves all PHY tasks
- not competitive with RZF precoding
- not a continual-learning result
- not an unseen-environment result
- not a sub-meter localization method

## Repository Structure

```text
wireless-fm-lora/
  README.md
  LICENSE
  requirements.txt
  environment.yml
  dataset_shapes_note.md
  configs/
    default.yaml
    proposed.yaml
    static_lora.yaml
    rzf_baseline.yaml
    ablation_A1_no_scene.yaml
    ablation_A2_no_adaptive_rank.yaml
    ablation_A3_no_hypernet.yaml
  scripts/
    generate_data.py
    train_multitask.py
    train_static_lora.py
    train_ablation.py
    eval_snr.py
    eval_localization_range.py
    eval_detection_debug.py
    run_rzf_baseline.py
  src/
    data/
    models/
    losses/
    metrics/
    utils/
  logs/
    Day11/
    Day12/
    Day13/
    OFFICIAL_NUMBERS.md
  results/
    checkpoints/
    tables/
    figures/
  docs/
    METHOD_MATCHES_CODE.md
    LOCKED_RESULTS.md
    HOW_TO_REPRODUCE.md
```

## Official results

| Model | Params (Trainable) | CE@-5dB | CE@25dB | PREC@25dB | LOC RMSE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Proposed | 35.09M (35,087,450) | −8.08 | −23.95 | 4.07 | 15.54 |
| Static LoRA | 8.62M (8,620,180) | −0.55 | −16.79 | 4.06 | 15.30 |
| A1 | 34.89M (34,890,586) | −1.33 | −23.83 | 3.98 | 13.62 |
| A2 | 35.04M (35,037,525) | −7.65 | −24.24 | 4.04 | 13.69 |
| A3 | 8.64M (8,640,789) | −1.14 | −11.87 | 4.91 | 12.27 |
| RZF | 0 | n/a | n/a | 103.44 | n/a |

*(All neural models use 124M frozen backbone parameters. The Proposed model features 35.09M trainable parameters).*

Detection BER = 0 at 20–25 dB is from MMSE initialization, not from Dynamic LoRA.

## Requirements

- Python >= 3.10
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- PEFT >= 0.4.0
- NumPy, SciPy, Pandas, Matplotlib, PyYAML

Installation:
```bash
pip install -r requirements.txt
```

## Reproduce the official experiment

### 1. Data Generation
```bash
python scripts/generate_data.py --env Urban_Macro --n 300 --seed 42
```

### 2. Model Training & Baselines
```bash
# Proposed Dynamic LoRA
python scripts/train_multitask.py --config configs/proposed.yaml

# Static LoRA (Fixed r=8)
python scripts/train_static_lora.py --config configs/static_lora.yaml

# Classical RZF Baseline
python scripts/run_rzf_baseline.py --config configs/rzf_baseline.yaml

# Architectural Ablations (Day 13)
python scripts/train_ablation.py --ablation A1_no_scene
python scripts/train_ablation.py --ablation A2_no_adaptive_rank
python scripts/train_ablation.py --ablation A3_no_hypernet
```

### 3. Checkpoint Evaluation (Part C6)
```bash
# Evaluate SNR sweep from trained checkpoint
python scripts/eval_snr.py --ckpt results/checkpoints/proposed_day\ 11_best.pt --out logs/Day11/proposed/snr_evaluation.csv

# Evaluate user localization range & RMSE
python scripts/eval_localization_range.py --ckpt results/checkpoints/proposed_day\ 11_best.pt

# Evaluate MIMO detection bit-level debug tables
python scripts/eval_detection_debug.py --ckpt results/checkpoints/proposed_day\ 11_best.pt --snr 20
python scripts/eval_detection_debug.py --ckpt results/checkpoints/proposed_day\ 11_best.pt --snr 25
```

## File map

Every locked official number maps to a specific local file in the repository:

- **Proposed Dynamic LoRA**:
  - SNR Metrics: [`logs/Day11/proposed/snr_evaluation.csv`](logs/Day11/proposed/snr_evaluation.csv)
  - Run Log: [`logs/Day11/proposed/run_log.txt`](logs/Day11/proposed/run_log.txt)
  - Training Epoch Progression: [`logs/Day11/proposed/metrics.csv`](logs/Day11/proposed/metrics.csv)
- **Static LoRA (r=8)**:
  - SNR Metrics: [`logs/Day11/static_lora/snr_evaluation.csv`](logs/Day11/static_lora/snr_evaluation.csv)
  - Run Log: [`logs/Day11/static_lora/run_log.txt`](logs/Day11/static_lora/run_log.txt)
- **Classical RZF Baseline**:
  - SNR Metrics: [`logs/Day11/rzf_baseline/snr_evaluation.csv`](logs/Day11/rzf_baseline/snr_evaluation.csv)
  - Run Log: [`logs/Day11/rzf_baseline/run_log.txt`](logs/Day11/rzf_baseline/run_log.txt)
- **Day 13 Ablations**:
  - A1 (No Scene): [`logs/Day13/A1_no_scene/snr_evaluation.csv`](logs/Day13/A1_no_scene/snr_evaluation.csv)
  - A2 (Fixed r=8): [`logs/Day13/A2_no_adaptive_rank/snr_evaluation.csv`](logs/Day13/A2_no_adaptive_rank/snr_evaluation.csv)
  - A3 (No Hypernet): [`logs/Day13/A3_no_hypernet/snr_evaluation.csv`](logs/Day13/A3_no_hypernet/snr_evaluation.csv)
- **Day 12 Verification Suites**:
  - Detection Bit Debug: [`logs/Day12/proposed_detection_debug.txt`](logs/Day12/proposed_detection_debug.txt) & [`.csv`](logs/Day12/proposed_detection_debug.csv)
  - Localization Range: [`logs/Day12/proposed_localization_range.txt`](logs/Day12/proposed_localization_range.txt) & [`.csv`](logs/Day12/proposed_localization_range.csv)
- **Official Summary References**:
  - [`logs/OFFICIAL_NUMBERS.md`](logs/OFFICIAL_NUMBERS.md)
  - [`docs/LOCKED_RESULTS.md`](docs/LOCKED_RESULTS.md)
  - [`docs/METHOD_MATCHES_CODE.md`](docs/METHOD_MATCHES_CODE.md)

## Citation / paper note

Results in README match the locked experimental record from Days 11–15. Do not replace them with unofficial tables.
