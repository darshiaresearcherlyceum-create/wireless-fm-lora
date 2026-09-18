import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any, Optional

class WirelessMultiTaskDataset(Dataset):
    """
    Unified Multi-Task Wireless Dataset for 5 Physical Layer Tasks:
      Task 0: Channel Estimation (Regression, NMSE in dB)
      Task 1: MIMO Detection (Classification/Regression, BER)
      Task 2: Multi-User Precoding (Regression, Sum-Rate in bps/Hz)
      Task 3: Channel Decoding (Binary Classification, BER/BLER)
      Task 4: User 3D Localization (Regression, RMSE in meters)
      
    High-performance implementation:
      - Pre-interpolates scene volume projections once at dataset initialization.
      - Vectorized C-level einsum for wireless OFDM signals.
      - Dynamic SNR noise synthesis.
    """
    def __init__(
        self,
        env_name: str,
        split: str = "train",
        data_dir: str = "dataset_wireless",
        train_split: float = 0.70,
        val_split: float = 0.15,
        max_samples: Optional[int] = None,
        snr_db: Optional[float] = None,
        seed: int = 42
    ):
        data_path = os.path.join(data_dir, f"{env_name}.npz")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file {data_path} not found. Run dataset generation first.")
            
        data = np.load(data_path)
        self.channel = data["channel"]                     # [N, K, N_BS, N_sub]
        self.precoding_matrix = data["precoding_matrix"]   # [N, N_BS, K, N_sub]
        self.transmitted_symbols = data["transmitted_symbols"] # [N, K, N_sub]
        self.transmitted_bits = data["transmitted_bits"]   # [N, K, N_sub, 2]
        self.user_positions = data["user_positions"]       # [N, K, 3]
        self.scene = data["scene"]                         # [N, 3, 64, 64]
        self.snr = data["snr"]                             # [N]
        
        # Pre-interpolate all scene projections at once (huge CPU speedup)
        scene_raw_all = torch.tensor(self.scene, dtype=torch.float32)
        self.scene_imgs = F.interpolate(
            scene_raw_all, size=(224, 224), mode="bilinear", align_corners=False
        ) # [N, 3, 224, 224]
        
        self.split = split
        self.snr_db = snr_db
        self.snr_candidates = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
        
        num_total = len(self.snr)
        np.random.seed(seed)
        shuffled = np.random.permutation(num_total)
        if max_samples is not None and max_samples < num_total:
            shuffled = shuffled[:max_samples]
            num_total = max_samples
            
        train_idx = int(train_split * num_total)
        val_idx = int((train_split + val_split) * num_total)
        
        if split == "train":
            self.indices = shuffled[:train_idx]
        elif split == "val":
            self.indices = shuffled[train_idx:val_idx]
        elif split == "test":
            self.indices = shuffled[val_idx:]
        else:
            raise ValueError(f"Invalid split: {split}")
            
    def set_snr(self, snr_db: Optional[float]):
        """Sets or clears fixed evaluation SNR."""
        self.snr_db = snr_db
            
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        real_idx = self.indices[idx]
        
        # 1. 3D Scene Projection (pre-interpolated)
        scene_img = self.scene_imgs[real_idx] # [3, 224, 224]
        
        # 2. Extract Base Physical Signals
        H = self.channel[real_idx]                 # [K, N_BS, N_sub]
        W = self.precoding_matrix[real_idx]        # [N_BS, K, N_sub]
        symbols = self.transmitted_symbols[real_idx] # [K, N_sub]
        bits = self.transmitted_bits[real_idx]     # [K, N_sub, 2]
        pos = self.user_positions[real_idx]        # [K, 3]
        
        K, N_BS, N_sub = H.shape
        
        # Determine SNR for this sample
        if self.snr_db is not None:
            curr_snr = float(self.snr_db)
        elif self.split == "train":
            curr_snr = float(np.random.choice(self.snr_candidates))
        else:
            curr_snr = float(self.snr[real_idx])
            
        noise_var = 10.0 ** (-curr_snr / 10.0)
        noise_std = np.sqrt(noise_var / 2.0)
        
        # 3. Fast Vectorized Noise Synthesis
        # Uplink Pilot: Y_p = H + N_p
        noise_p = np.random.normal(0, noise_std, size=H.shape) + 1j * np.random.normal(0, noise_std, size=H.shape)
        Y_p = H + noise_p
        
        # Downlink Multi-User Data: Y_d = H * W * X + N_d (Vectorized einsum)
        sig = np.einsum('kbm, bjm, jm -> km', H, W, symbols) # [K, N_sub]
        noise_d = np.random.normal(0, noise_std, size=sig.shape) + 1j * np.random.normal(0, noise_std, size=sig.shape)
        received_data = sig + noise_d
        
        # 4. Construct Task Inputs and Targets
        # --- Task 0: Channel Estimation ---
        Y_p_real, Y_p_imag = np.real(Y_p), np.imag(Y_p)
        task0_in = np.concatenate([Y_p_real, Y_p_imag], axis=0).transpose(2, 0, 1).reshape(N_sub, -1) # [N_sub, 1024]
        
        H_real, H_imag = np.real(H), np.imag(H)
        target0 = np.concatenate([H_real, H_imag], axis=0).transpose(2, 0, 1).reshape(N_sub, -1)     # [N_sub, 1024]
        
        # --- Task 1: MIMO Detection ---
        Y_d_real, Y_d_imag = np.real(received_data), np.imag(received_data)
        yd_all = np.concatenate([Y_d_real, Y_d_imag], axis=0).transpose(1, 0) # [N_sub, 16]
        h_all = target0 # [N_sub, 1024]
        task1_in = np.concatenate([yd_all, h_all], axis=1) # [N_sub, 1040]
        
        syms_real, syms_imag = np.real(symbols), np.imag(symbols)
        target1 = np.concatenate([syms_real, syms_imag], axis=0).transpose(1, 0) # [N_sub, 16]
        
        # --- Task 2: Precoding (Pilot channel observations, not ground-truth channel) ---
        task2_in = task0_in
        W_real, W_imag = np.real(W), np.imag(W)
        target2 = np.concatenate([W_real, W_imag], axis=0).transpose(2, 0, 1).reshape(N_sub, -1)     # [N_sub, 1024]
        
        # --- Task 3: Channel Decoding ---
        task3_in = yd_all # [N_sub, 16]
        target3 = bits.transpose(1, 0, 2).reshape(N_sub, -1) # [N_sub, 16]
        
        # --- Task 4: User Localization (Noisy Uplink Pilots Y_p + Data Signals) ---
        task4_in = np.concatenate([yd_all, task0_in], axis=1) # [N_sub, 1040]
        target4 = (pos / 100.0).flatten() # [24] normalized in [0, 1] for 100m area
        
        # 5. Pad inputs to uniform 1040 dimension
        t0_pad = np.pad(task0_in, ((0, 0), (0, 16)), mode='constant')
        t1_pad = task1_in
        t2_pad = np.pad(task2_in, ((0, 0), (0, 16)), mode='constant')
        t3_pad = np.pad(task3_in, ((0, 0), (0, 1024)), mode='constant')
        t4_pad = task4_in
        
        inputs_stacked = np.stack([t0_pad, t1_pad, t2_pad, t3_pad, t4_pad], axis=0) # [5, N_sub, 1040]
        
        return {
            'scene': scene_img,
            'inputs': torch.tensor(inputs_stacked, dtype=torch.float32),
            'target_0': torch.tensor(target0, dtype=torch.float32),
            'target_1': torch.tensor(target1, dtype=torch.float32),
            'target_2': torch.tensor(target2, dtype=torch.float32),
            'target_3': torch.tensor(target3, dtype=torch.float32),
            'target_4': torch.tensor(target4, dtype=torch.float32),
            'snr': torch.tensor(curr_snr, dtype=torch.float32)
        }

def get_dataloaders(
    env_name: str,
    data_dir: str = "dataset_wireless",
    batch_size: int = 16,
    train_split: float = 0.70,
    val_split: float = 0.15,
    max_samples: Optional[int] = None,
    snr_db: Optional[float] = None,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = WirelessMultiTaskDataset(env_name, "train", data_dir, train_split, val_split, max_samples, snr_db=None, seed=seed)
    val_ds = WirelessMultiTaskDataset(env_name, "val", data_dir, train_split, val_split, max_samples, snr_db=snr_db, seed=seed)
    test_ds = WirelessMultiTaskDataset(env_name, "test", data_dir, train_split, val_split, max_samples, snr_db=snr_db, seed=seed)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
