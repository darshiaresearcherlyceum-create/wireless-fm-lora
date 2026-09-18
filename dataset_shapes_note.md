# Dataset Verification & Shapes Note (Urban_Macro)

**Generation Command:**  
```bash
python scripts/generate_data.py --samples 300 --env Urban_Macro --seed 42
```

**Dataset File:** [`dataset_wireless/Urban_Macro.npz`](dataset_wireless/Urban_Macro.npz)  
**Propagation Model:** 3GPP TR 38.901 Urban Macro (UMa) Clustered Delay Line (CDL) & Spatial Channel Model

---

## 1. Official Dataset Split

| Split | Ratio | Number of Samples ($N$) | Purpose |
| :--- | :---: | :---: | :--- |
| **Train** | 70% | 210 | Multi-task model and baseline training |
| **Validation** | 15% | 45 | Checkpoint selection and hyperparameter monitoring |
| **Test** | 15% | 45 | Final locked SNR-sweep evaluation ($-5$ to $25$ dB) |
| **Total** | 100% | 300 | Single official environment dataset |

---

## 2. Array Shapes & Data Types

The file `dataset_wireless/Urban_Macro.npz` contains 300 samples with $K = 8$ users, $N_{\text{BS}} = 64$ base station antennas ($8 \times 8$ UPA), and $N_{\text{sub}} = 64$ OFDM subcarriers:

| Key Name | Symbol | Shape ($N, \dots$) | Data Type | Physical Description |
| :--- | :---: | :---: | :---: | :--- |
| `channel` | $\mathbf{H}$ | `(300, 8, 64, 64)` | `complex64` | Complex frequency-domain MIMO channel ($K \times N_{\text{BS}} \times N_{\text{sub}}$) |
| `received_pilot` | $\mathbf{Y}_p$ | `(300, 8, 64, 64)` | `complex64` | Uplink pilot observations with AWGN noise |
| `user_positions` | $[x, y, z]$ | `(300, 8, 3)` | `float32` | Ground-truth 3D user coordinates in meters ($100\text{ m} \times 100\text{ m}$ arena) |
| `transmitted_bits` | $\mathbf{b}$ | `(300, 8, 64, 2)` | `int8` | Ground-truth binary bits (2 bits per QPSK symbol) |
| `received_data` | $\mathbf{Y}_d$ | `(300, 8, 64)` | `complex64` | Downlink received multi-user data symbols |
| `transmitted_symbols` | $\mathbf{X}$ | `(300, 8, 64)` | `complex64` | Transmitted unit-energy QPSK constellation symbols |
| `precoding_matrix` | $\mathbf{W}$ | `(300, 64, 8, 64)` | `complex64` | Downlink beamforming matrix ($N_{\text{BS}} \times K \times N_{\text{sub}}$) |
| `scene` | $\mathcal{S}$ | `(300, 3, 64, 64)` | `float32` | 3D orthographic occupancy projections (Top, Front, Side views) |
| `snr` | $\gamma_{\text{dB}}$ | `(300,)` | `float32` | Assigned operating SNR in dB ($-5$ to $25$ dB) |

---

## 3. Verification & Integrity

- **Environment Scope:** Strictly restricted to `Urban_Macro.npz` matching the paper setup.
- **Physical Consistency:** Unit channel power normalization, orthogonal 3GPP UPA steering vectors, and non-collapsed user positions.
- **Configuration Match:** Matches the official settings specified in [`configs/proposed.yaml`](configs/proposed.yaml) and [`configs/default.yaml`](configs/default.yaml).
