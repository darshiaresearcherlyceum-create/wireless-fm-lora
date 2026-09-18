import numpy as np
import os
import tqdm
from typing import Optional

import numpy as np
import os
import tqdm
from typing import Optional

def generate_channel_dataset(
    env_name: str,
    output_dir: str = "dataset_wireless",
    num_samples: int = 500,
    num_users: int = 8,
    num_subcarriers: int = 64,
    num_bs_antennas: int = 64,
    carrier_freq_hz: float = 3.5e9,
    subcarrier_spacing_hz: float = 15e3,
    seed: int = 42
) -> str:
    """
    Generates a realistic wireless dataset representing a specific propagation environment
    following 3GPP TR 38.901 (RMa, UMa, InF) Clustered Delay Line (CDL) & Spatial Channel Models.
    
    Generates:
      1. Spatial 3D Scene Projections (Top, Front, Side view occupancy grid).
      2. Complex MIMO-OFDM Channel matrices H.
      3. Uplink Pilot Observations Y_p for Channel Estimation.
      4. Downlink Multi-user Data Signals Y_d.
      5. Zero-Forcing (ZF) Beamforming/Precoding matrices W.
      6. Transmitted QPSK symbols and raw information bits.
      7. Ground-truth 3D User positions for localization.
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    Ny = int(np.sqrt(num_bs_antennas))
    Nz = num_bs_antennas // Ny
    assert Ny * Nz == num_bs_antennas, "num_bs_antennas must be factorizable into 2D UPA"
    
    # 3GPP TR 38.901 Environment propagation characteristics
    fc_ghz = carrier_freq_hz / 1e9  # 3.5 GHz
    
    if env_name == "Rural_Macro":
        # 3GPP RMa: Dominant LoS, sparse scattering, long delay spread ~10-30 ns
        num_clusters = 2
        num_paths = 2
        delay_spread = 20e-9  # 20 ns
        scene_occupancy_rate = 0.05
        pl_const = 32.4 + 20.0 * np.log10(fc_ghz)
        pl_exponent = 2.0
    elif env_name == "Urban_Macro":
        # 3GPP UMa: Moderate scattering, multipath clusters, delay spread ~50-80 ns
        num_clusters = 5
        num_paths = 3
        delay_spread = 60e-9  # 60 ns
        scene_occupancy_rate = 0.20
        pl_const = 28.0 + 20.0 * np.log10(fc_ghz)
        pl_exponent = 2.2
    elif env_name == "Indoor_Factory":
        # 3GPP InF: Rich multipath from metal structures, high delay spread ~150-250 ns
        num_clusters = 8
        num_paths = 4
        delay_spread = 180e-9  # 180 ns
        scene_occupancy_rate = 0.40
        pl_const = 31.84 + 19.0 * np.log10(fc_ghz)
        pl_exponent = 2.15
    elif env_name == "Unseen_Urban_Dense":
        # Dense Urban High-rise: Extreme multipath & shadowing
        num_clusters = 10
        num_paths = 4
        delay_spread = 120e-9  # 120 ns
        scene_occupancy_rate = 0.35
        pl_const = 30.0 + 20.0 * np.log10(fc_ghz)
        pl_exponent = 2.4
    else:
        raise ValueError(f"Unknown environment name: {env_name}")
        
    df = subcarrier_spacing_hz
    bs_pos = np.array([50.0, 50.0, 20.0]) # Base Station at [50m, 50m, 20m]
    
    channels_all = []
    received_pilots_all = []
    received_data_all = []
    transmitted_bits_all = []
    transmitted_symbols_all = []
    precoding_matrices_all = []
    user_positions_all = []
    scenes_all = []
    snrs_all = []
    
    for _ in tqdm.trange(num_samples, desc=f"Generating 3GPP {env_name}"):
        # 1. 3D Scene Representation: 3-channel Orthographic Projections (Top, Front, Side) of 64x64x64 volume
        top_view = np.zeros((64, 64), dtype=np.float32)
        front_view = np.zeros((64, 64), dtype=np.float32)
        side_view = np.zeros((64, 64), dtype=np.float32)
        
        # Place Base Station in scene
        bs_idx = np.clip((bs_pos / 100.0 * 63).astype(int), 0, 63)
        top_view[bs_idx[0], bs_idx[1]] = 1.0
        front_view[bs_idx[0], bs_idx[2]] = 1.0
        side_view[bs_idx[1], bs_idx[2]] = 1.0
        
        # Obstacles / scatterers
        num_obstacles = int(scene_occupancy_rate * 50)
        for _ in range(num_obstacles):
            obs_start = np.random.randint(0, 50, size=3)
            obs_size = np.random.randint(2, 15, size=3)
            top_view[obs_start[0]:obs_start[0]+obs_size[0], obs_start[1]:obs_start[1]+obs_size[1]] = 1.0
            front_view[obs_start[0]:obs_start[0]+obs_size[0], obs_start[2]:obs_start[2]+obs_size[2]] = 1.0
            side_view[obs_start[1]:obs_start[1]+obs_size[1], obs_start[2]:obs_start[2]+obs_size[2]] = 1.0
            
        # 2. User Coordinates (K users within 100m x 100m area)
        user_pos = np.random.uniform(10.0, 90.0, size=(num_users, 3))
        user_pos[:, 2] = np.random.uniform(1.5, 3.0, size=num_users) # pedestrian height
        
        for k in range(num_users):
            u_idx = np.clip((user_pos[k] / 100.0 * 63).astype(int), 0, 63)
            top_view[u_idx[0], u_idx[1]] = 1.0
            front_view[u_idx[0], u_idx[2]] = 1.0
            side_view[u_idx[1], u_idx[2]] = 1.0
            
        scene_img = np.stack([top_view, front_view, side_view], axis=0) # [3, 64, 64]
        
        # 3. 3GPP Spatial Channel Model H: [K, N_BS, N_sub]
        H_raw = np.zeros((num_users, num_bs_antennas, num_subcarriers), dtype=complex)
        for k in range(num_users):
            dx = float(user_pos[k, 0] - bs_pos[0])
            dy = float(user_pos[k, 1] - bs_pos[1])
            dz = float(user_pos[k, 2] - bs_pos[2])
            dist_3d = float(np.sqrt(dx**2 + dy**2 + dz**2))
            
            # Geometric True AoA (Azimuth and Elevation) relative to BS
            theta_los = float(np.arctan2(dy, dx))
            phi_los = float(np.arccos(np.clip(-dz / max(dist_3d, 1e-6), -1.0, 1.0)))
            
            # 3GPP Path Loss in dB (with fc in GHz)
            path_loss_db = pl_const + 10.0 * pl_exponent * np.log10(dist_3d)
            shadowing_db = np.random.normal(0, 4.0)
            total_loss_linear = 10.0 ** (-(path_loss_db + shadowing_db) / 20.0)
            
            for c in range(num_clusters):
                # Cluster central angles around true geometric AoA
                if c == 0:
                    theta_c = theta_los
                    phi_c = phi_los
                else:
                    theta_c = theta_los + np.random.laplace(0, 0.25)
                    phi_c = np.clip(phi_los + np.random.laplace(0, 0.15), 0.1 * np.pi, 0.9 * np.pi)
                    
                for p in range(num_paths):
                    # Ray angle offsets
                    theta_cp = theta_c + np.random.laplace(0, 0.05)
                    phi_cp = phi_c + np.random.laplace(0, 0.03)
                    tau_cp = (0.0 if c == 0 and p == 0 else np.random.exponential(delay_spread))
                    
                    ray_gain = (np.random.normal(0, 1) + 1j * np.random.normal(0, 1)) / np.sqrt(2)
                    if c == 0 and p == 0:
                        ray_gain = (ray_gain + 2.0) / np.sqrt(5.0) # Dominant LoS
                    ray_gain *= total_loss_linear / np.sqrt(num_clusters * num_paths)
                    
                    # 2D UPA Steering Vector
                    steer_y = np.exp(1j * np.pi * np.arange(Ny) * np.sin(theta_cp) * np.sin(phi_cp))
                    steer_z = np.exp(1j * np.pi * np.arange(Nz) * np.cos(theta_cp))
                    steer = np.kron(steer_y, steer_z) # [N_BS]
                    
                    # Multi-carrier frequency response
                    phase_freq = np.exp(-1j * 2 * np.pi * np.arange(num_subcarriers) * tau_cp * df)
                    H_raw[k, :, :] += ray_gain * np.outer(steer, phase_freq)
                    
        # Power normalization so average channel power per subcarrier is unit-scale
        ch_power = np.mean(np.abs(H_raw)**2) + 1e-12
        H = H_raw / np.sqrt(ch_power)
        
        # 4. Transmitted Bits and QPSK Symbols
        bits = np.random.randint(0, 2, size=(num_users, num_subcarriers, 2))
        symbols = ((1.0 - 2.0 * bits[:, :, 0]) + 1j * (1.0 - 2.0 * bits[:, :, 1])) / np.sqrt(2)
        
        # 5. Zero-Forcing (ZF) Precoding Matrix W: [N_BS, K, N_sub]
        W = np.zeros((num_bs_antennas, num_users, num_subcarriers), dtype=complex)
        for m in range(num_subcarriers):
            Hm = H[:, :, m] # [K, N_BS]
            Hm_pinv = np.linalg.pinv(Hm) # [N_BS, K]
            col_norm = np.linalg.norm(Hm_pinv, axis=0, keepdims=True) + 1e-6
            W[:, :, m] = Hm_pinv / col_norm
            
        # 6. AWGN Channel & Multi-Task Observations
        snr_db = float(np.random.choice([-5, 0, 5, 10, 15, 20, 25]))
        noise_var = 10**(-snr_db / 10.0)
        
        # Uplink Pilots Y_p = H + Noise
        noise_p = (np.random.normal(0, np.sqrt(noise_var / 2.0), size=H.shape) +
                   1j * np.random.normal(0, np.sqrt(noise_var / 2.0), size=H.shape))
        received_pilot = H + noise_p
        
        # Downlink Data Y_d = H * W * X + Noise
        received_data = np.zeros((num_users, num_subcarriers), dtype=complex)
        for m in range(num_subcarriers):
            Hm = H[:, :, m]
            Wm = W[:, :, m]
            Xm = symbols[:, m]
            sig = Hm @ Wm @ Xm
            noise_d = (np.random.normal(0, np.sqrt(noise_var / 2.0), size=num_users) +
                       1j * np.random.normal(0, np.sqrt(noise_var / 2.0), size=num_users))
            received_data[:, m] = sig + noise_d
            
        channels_all.append(H)
        received_pilots_all.append(received_pilot)
        received_data_all.append(received_data)
        transmitted_bits_all.append(bits)
        transmitted_symbols_all.append(symbols)
        precoding_matrices_all.append(W)
        user_positions_all.append(user_pos)
        scenes_all.append(scene_img)
        snrs_all.append(snr_db)
        
    save_path = os.path.join(output_dir, f"{env_name}.npz")
    np.savez(
        save_path,
        channel=np.array(channels_all, dtype=np.complex64),
        received_pilot=np.array(received_pilots_all, dtype=np.complex64),
        received_data=np.array(received_data_all, dtype=np.complex64),
        transmitted_bits=np.array(transmitted_bits_all, dtype=np.int8),
        transmitted_symbols=np.array(transmitted_symbols_all, dtype=np.complex64),
        precoding_matrix=np.array(precoding_matrices_all, dtype=np.complex64),
        user_positions=np.array(user_positions_all, dtype=np.float32),
        scene=np.array(scenes_all, dtype=np.float32),
        snr=np.array(snrs_all, dtype=np.float32)
    )
    return save_path
