# Official Locked Benchmark Results

This document records the official, locked experimental numbers for the paper, cross-referenced with local repository source files.

---

## 1. Locked Benchmark Table (Urban_Macro)

| Model | Params | CE@-5dB | CE@25dB | PREC@25dB | LOC RMSE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Proposed** | 35,087,450 | −8.08 | −23.95 | 4.07 | 15.54 |
| **Static LoRA** | 8,620,180 | −0.55 | −16.79 | 4.06 | 15.30 |
| **A1 (No Scene)** | 34,890,586 | −1.33 | −23.83 | 3.98 | 13.62 |
| **A2 (No Adaptive Rank)** | 35,037,525 | −7.65 | −24.24 | 4.04 | 13.69 |
| **A3 (No Hypernet)** | 8,640,789 | −1.14 | −11.87 | 4.91 | 12.27 |
| **RZF (Classical Baseline)** | 0 | n/a | n/a | 103.44 | n/a |

> **Detection Performance Attribution Note:**  
> *Detection BER = 0 at 20–25 dB is from MMSE initialization, not from Dynamic LoRA.*

---

## 2. Audit of Local Data Sources

| Model / Baseline | Evaluation File Path | Training Log Path |
| :--- | :--- | :--- |
| **Proposed Dynamic LoRA** | [`logs/Day11/proposed/snr_evaluation.csv`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day11/proposed/snr_evaluation.csv) | [`logs/Day11/proposed/run_log.txt`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day11/proposed/run_log.txt) |
| **Static LoRA (r=8)** | [`logs/Day11/static_lora/snr_evaluation.csv`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day11/static_lora/snr_evaluation.csv) | [`logs/Day11/static_lora/run_log.txt`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day11/static_lora/run_log.txt) |
| **Classical RZF Baseline** | [`logs/Day11/rzf_baseline/snr_evaluation.csv`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day11/rzf_baseline/snr_evaluation.csv) | [`logs/Day11/rzf_baseline/run_log.txt`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day11/rzf_baseline/run_log.txt) |
| **Ablation A1** | [`logs/Day13/A1_no_scene/snr_evaluation.csv`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day13/A1_no_scene/snr_evaluation.csv) | [`logs/Day13/A1_no_scene/run_log.txt`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day13/A1_no_scene/run_log.txt) |
| **Ablation A2** | [`logs/Day13/A2_no_adaptive_rank/snr_evaluation.csv`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day13/A2_no_adaptive_rank/snr_evaluation.csv) | [`logs/Day13/A2_no_adaptive_rank/run_log.txt`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day13/A2_no_adaptive_rank/run_log.txt) |
| **Ablation A3** | [`logs/Day13/A3_no_hypernet/snr_evaluation.csv`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day13/A3_no_hypernet/snr_evaluation.csv) | [`logs/Day13/A3_no_hypernet/run_log.txt`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day13/A3_no_hypernet/run_log.txt) |
| **Consolidated Benchmark Table** | [`results/tables/benchmark_comparison_Day 11.csv`](file:///c:/Users/resea/Downloads/wireless-fm-lora/results/tables/benchmark_comparison_Day%2011.csv) | N/A |

---

## 3. Localization Statistical Range & Breakdown

Source file: [`logs/Day12/proposed_localization_range.csv`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day12/proposed_localization_range.csv)

| Coordinate | Pred Min (m) | Pred Max (m) | True Min (m) | True Max (m) | Test RMSE (m) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **X** | 17.48 | 81.42 | 10.18 | 89.97 | 12.63 |
| **Y** | 12.94 | 84.62 | 10.04 | 89.97 | 9.02 |
| **Z** | 1.30 | 7.12 | 1.50 | 3.00 | 0.78 |
| **Overall 3D** | — | — | — | — | **15.54 m** |

---

## 4. MIMO Detection Debug Breakdown (at 20 & 25 dB SNR)

Source file: [`logs/Day12/proposed_detection_debug.txt`](file:///c:/Users/resea/Downloads/wireless-fm-lora/logs/Day12/proposed_detection_debug.txt)

- **Total bits evaluated**: 46,080 bits ($45\text{ test samples} \times 64\text{ subcarriers} \times 8\text{ users} \times 2\text{ bits}$).
- **Bit Errors at 20 dB**: 0 (BER = 0.00000000).
- **Bit Errors at 25 dB**: 0 (BER = 0.00000000).
- **Initialization**: Linear MMSE base estimation provides near error-free symbol anchors at high SNR.
