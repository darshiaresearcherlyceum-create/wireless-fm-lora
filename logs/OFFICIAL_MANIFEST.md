# Official Authoritative Benchmark Artifacts

The following experimental logs, debug suites, and ablation runs are marked as **OFFICIAL** and authoritative:

---

### 1. Day 11: Core Methods Comparison (Urban_Macro)
- **Proposed Dynamic LoRA**:
  - Log: [`logs/Day11/proposed/run_log.txt`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day11/proposed/run_log.txt)
  - SNR Evaluation: [`logs/Day11/proposed/snr_evaluation.csv`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day11/proposed/snr_evaluation.csv)
  - Training Metrics: [`logs/Day11/proposed/metrics.csv`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day11/proposed/metrics.csv)
  - Config: [`logs/Day11/proposed/config.yaml`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day11/proposed/config.yaml)
- **Static LoRA (Rank 8 Baseline)**:
  - Log: [`logs/Day11/static_lora/run_log.txt`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day11/static_lora/run_log.txt)
  - SNR Evaluation: [`logs/Day11/static_lora/snr_evaluation.csv`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day11/static_lora/snr_evaluation.csv)
  - Training Metrics: [`logs/Day11/static_lora/metrics.csv`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day11/static_lora/metrics.csv)
- **Classical RZF Precoder Baseline**:
  - Log: [`logs/Day11/rzf_baseline/run_log.txt`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day11/rzf_baseline/run_log.txt)
  - SNR Evaluation: [`logs/Day11/rzf_baseline/snr_evaluation.csv`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day11/rzf_baseline/snr_evaluation.csv)
- **Full Consolidated Benchmark Table**:
  - [`results/tables/benchmark_comparison_Day 11.csv`](file:///c:/Users/resea/Downloads/wireless-fm-lora/results/tables/benchmark_comparison_Day%2011.csv)

---

### 2. Day 12: Diagnosis & Verification Suite
- **Localization Spatial Range Verification**:
  - Proposed: [`logs/Day12/proposed_localization_range.csv`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day12/proposed_localization_range.csv) & [`logs/Day12/proposed_localization_examples.txt`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day12/proposed_localization_examples.txt)
  - Static LoRA: [`logs/Day12/static_lora_localization_range.csv`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day12/static_lora_localization_range.csv) & [`logs/Day12/static_lora_localization_examples.txt`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day12/static_lora_localization_examples.txt)
- **MIMO Detection Bit-Level Debug Table**:
  - Proposed: [`logs/Day12/proposed_detection_debug.csv`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day12/proposed_detection_debug.csv) & [`logs/Day12/proposed_detection_debug.txt`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day12/proposed_detection_debug.txt)
  - Static LoRA: [`logs/Day12/static_lora_detection_debug.csv`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day12/static_lora_detection_debug.csv) & [`logs/Day12/static_lora_detection_debug.txt`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day12/static_lora_detection_debug.txt)

---

### 3. Day 13: Architecture Ablation Studies
- **A1: Ablation without Scene Conditioning**:
  - [`logs/Day13/A1_no_scene/run_log.txt`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day13/A1_no_scene/run_log.txt)
  - [`logs/Day13/A1_no_scene/snr_evaluation.csv`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day13/A1_no_scene/snr_evaluation.csv)
  - [`logs/Day13/A1_no_scene/metrics.csv`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day13/A1_no_scene/metrics.csv)
- **A2: Ablation without Adaptive Rank**:
  - [`logs/Day13/A2_no_adaptive_rank/run_log.txt`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day13/A2_no_adaptive_rank/run_log.txt)
  - [`logs/Day13/A2_no_adaptive_rank/snr_evaluation.csv`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day13/A2_no_adaptive_rank/snr_evaluation.csv)
  - [`logs/Day13/A2_no_adaptive_rank/metrics.csv`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day13/A2_no_adaptive_rank/metrics.csv)
- **A3: Ablation without Hypernetwork**:
  - [`logs/Day13/A3_no_hypernet/run_log.txt`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day13/A3_no_hypernet/run_log.txt)
  - [`logs/Day13/A3_no_hypernet/snr_evaluation.csv`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day13/A3_no_hypernet/snr_evaluation.csv)
  - [`logs/Day13/A3_no_hypernet/metrics.csv`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day13/A3_no_hypernet/metrics.csv)

---

Official locked reference summary: [`logs/OFFICIAL_NUMBERS.md`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/OFFICIAL_NUMBERS.md)
