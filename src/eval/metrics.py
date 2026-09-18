import numpy as np
import torch
from typing import Tuple, Dict

def evaluate_channel_estimation_nmse(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Computes Normalized Mean Square Error (NMSE in dB) for Channel Estimation (Task 0).
    preds, targets: shape [B, N_sub, 1024] or [B, ...]
    """
    num = np.sum((targets - preds) ** 2)
    denom = np.sum(targets ** 2)
    nmse_lin = num / (denom + 1e-12)
    nmse_lin = max(float(nmse_lin), 1e-12)
    return float(10.0 * np.log10(nmse_lin))

def evaluate_mimo_detection_nmse(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Computes Normalized Mean Square Error (NMSE in dB) for MIMO Detection (Task 1).
    preds, targets: shape [B, N_sub, 16] or [B, ...]
    """
    num = np.sum((targets - preds) ** 2)
    denom = np.sum(targets ** 2)
    nmse_lin = num / (denom + 1e-12)
    nmse_lin = max(float(nmse_lin), 1e-12)
    return float(10.0 * np.log10(nmse_lin))

def evaluate_mimo_detection_ber(preds: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    """
    Computes Bit Error Rate (BER) and Symbol Error Rate (SER) for MIMO Detection (Task 1).
    preds: [B, N_sub, 16] (Continuous QPSK symbol predictions)
    targets: [B, N_sub, 16] (Ground truth QPSK symbols or binary bits)
    """
    # QPSK Demodulation: positive real/imag -> bit 0, negative real/imag -> bit 1
    detected_bits = (preds < 0.0).astype(int)
    
    # Check if targets are continuous constellation symbols (mean magnitude near 0.707) or binary bits {0, 1}
    if np.issubdtype(targets.dtype, np.floating) and np.any(targets < 0.0):
        target_bits = (targets < 0.0).astype(int)
    else:
        target_bits = (targets > 0.5).astype(int)
        
    num_bit_errors = np.sum(detected_bits != target_bits)
    total_bits = target_bits.size
    ber = float(num_bit_errors / total_bits)
    
    # SER: error if either of the 2 bits in a QPSK symbol is incorrect
    det_sym = detected_bits.reshape(-1, 2)
    tgt_sym = target_bits.reshape(-1, 2)
    num_sym_errors = np.sum(np.any(det_sym != tgt_sym, axis=1))
    ser = float(num_sym_errors / det_sym.shape[0])
    
    return ber, ser

def evaluate_precoding_sum_rate(preds: np.ndarray, true_channels: np.ndarray, snr_db: float = 10.0) -> float:
    """
    Computes Downlink Multi-User Achievable Sum-Rate (bits/s/Hz) for Precoding (Task 2).
    preds: [B, N_sub, 1024] - predicted beamforming matrix W
    true_channels: [B, N_sub, 1024] - ground truth channel matrix H
    """
    B_size = preds.shape[0]
    N_sub, K, N_BS = 64, 8, 64
    
    ch_real = true_channels[:, :, :512].reshape(B_size, N_sub, K, N_BS).transpose(0, 2, 3, 1)
    ch_imag = true_channels[:, :, 512:].reshape(B_size, N_sub, K, N_BS).transpose(0, 2, 3, 1)
    H = ch_real + 1j * ch_imag # [B, K, N_BS, N_sub]
    
    prec_real = preds[:, :, :512].reshape(B_size, N_sub, N_BS, K).transpose(0, 2, 3, 1)
    prec_imag = preds[:, :, 512:].reshape(B_size, N_sub, N_BS, K).transpose(0, 2, 3, 1)
    W = prec_real + 1j * prec_imag # [B, N_BS, K, N_sub]
    
    noise_power = 10**(-snr_db / 10.0)
    
    # Transpose for vectorized batch matrix multiplication
    # H: [B, N_sub, K, N_BS], W: [B, N_sub, N_BS, K]
    H_mat = H.transpose(0, 3, 1, 2)
    W_mat = W.transpose(0, 3, 1, 2)
    
    # Normalize W columns (BS antennas dimension)
    W_norm = np.linalg.norm(W_mat, axis=2, keepdims=True) + 1e-6
    W_mat = W_mat / W_norm
    
    # Effective Multi-User Channel: [B, N_sub, K, K]
    HW = np.matmul(H_mat, W_mat)
    HW_sq = np.abs(HW)**2
    
    # Diagonal is desired signal power [B, N_sub, K]
    signal = np.diagonal(HW_sq, axis1=2, axis2=3)
    total_power = np.sum(HW_sq, axis=3) # Sum over interferers
    interference = total_power - signal
    
    sinr = signal / (interference + noise_power + 1e-12)
    user_rates = np.log2(1.0 + sinr) # [B, N_sub, K]
    sum_rate_per_subcarrier = np.sum(user_rates, axis=-1) # [B, N_sub]
    avg_sum_rate = np.mean(sum_rate_per_subcarrier)
    
    return float(avg_sum_rate)

def compute_classical_rzf_precoder(channels: np.ndarray, alpha: float = 0.018) -> np.ndarray:
    """
    Computes analytical Regularized Zero-Forcing (RZF) precoding matrix W from channel H.
    channels: [B, N_sub, 1024]
    Returns: [B, N_sub, 1024] (flattened real/imag precoder)
    """
    B_size = channels.shape[0]
    N_sub, K, N_BS = 64, 8, 64
    ch_real = channels[:, :, :512].reshape(B_size, N_sub, K, N_BS)
    ch_imag = channels[:, :, 512:].reshape(B_size, N_sub, K, N_BS)
    H_c = ch_real + 1j * ch_imag # [B, N_sub, K, N_BS]
    
    # HH: [B, N_sub, K, K]
    HH = H_c @ H_c.transpose(0, 1, 3, 2).conj()
    reg = alpha * np.eye(K, dtype=complex)[None, None, :, :]
    inv = np.linalg.inv(HH + reg) # [B, N_sub, K, K]
    W_rzf = H_c.transpose(0, 1, 3, 2).conj() @ inv # [B, N_sub, N_BS, K]
    
    # Normalize columns
    w_norm = np.linalg.norm(W_rzf, axis=2, keepdims=True) + 1e-8
    W_norm = W_rzf / w_norm
    
    W_flat = np.concatenate([
        np.real(W_norm).reshape(B_size, N_sub, 512),
        np.imag(W_norm).reshape(B_size, N_sub, 512)
    ], axis=-1)
    return W_flat

def compute_classical_zf_precoder(channels: np.ndarray) -> np.ndarray:
    """
    Computes analytical Zero-Forcing (ZF) precoding matrix W from channel H (alpha ~ 0).
    channels: [B, N_sub, 1024]
    Returns: [B, N_sub, 1024]
    """
    return compute_classical_rzf_precoder(channels, alpha=1e-6)

def evaluate_channel_decoding(preds: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    """
    Computes Bit Error Rate (BER) and Block Error Rate (BLER) for Channel Decoding (Task 3).
    preds: [B, N_sub, 16]
    targets: [B, N_sub, 16]
    """
    B_size = preds.shape[0]
    decoded_bits = (preds > 0.0).astype(float)
    
    num_bit_errors = np.sum(decoded_bits != targets)
    ber = float(num_bit_errors / targets.size)
    
    block_errors = 0
    for b in range(B_size):
        if np.any(decoded_bits[b] != targets[b]):
            block_errors += 1
    bler = float(block_errors / B_size)
    
    return ber, bler

def evaluate_user_localization_rmse(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Computes Root Mean Square Error (RMSE in meters) for User Localization (Task 4).
    preds: [B, 24] or [B, K, 3]
    targets: [B, 24] or [B, K, 3]
    """
    B_size = preds.shape[0]
    p_est = preds.reshape(B_size, -1, 3)
    p_true = targets.reshape(B_size, -1, 3)
    
    # Check if coordinates are normalized in [0, 1] range; if so, scale to 100m area
    if np.max(np.abs(p_true)) <= 1.05:
        p_est = p_est * 100.0
        p_true = p_true * 100.0
        
    sq_err = np.sum((p_true - p_est)**2, axis=-1) # [B, K] sum of squared x,y,z errors
    rmse = np.sqrt(np.mean(sq_err)) # overall RMSE in meters across all users & batch
    return float(rmse)

def evaluate_user_localization_error_distance(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Computes Mean Euclidean Error Distance (in meters) for User Localization (Task 4).
    preds: [B, 24] or [B, K, 3]
    targets: [B, 24] or [B, K, 3]
    """
    B_size = preds.shape[0]
    p_est = preds.reshape(B_size, -1, 3)
    p_true = targets.reshape(B_size, -1, 3)
    
    # Check if coordinates are normalized in [0, 1] range; if so, scale to 100m area
    if np.max(np.abs(p_true)) <= 1.05:
        p_est = p_est * 100.0
        p_true = p_true * 100.0
        
    dist_err = np.sqrt(np.sum((p_true - p_est)**2, axis=-1)) # [B, K] Euclidean distance error per user
    return float(np.mean(dist_err)) # Mean Error Distance in meters

def compute_continual_learning_metrics(R: np.ndarray) -> Dict[str, float]:
    """
    Computes Continual Learning Metrics from performance matrix R where R[i, j]
    is the performance on environment j after learning environment i.
    Returns ACC (Average Accuracy), BWT (Backward Transfer / Forgetting), FWT (Forward Transfer).
    """
    S = R.shape[0]
    acc = float(np.mean(R[-1, :]))
    
    bwt = 0.0
    if S > 1:
        diffs = [R[-1, j] - R[j, j] for j in range(S - 1)]
        bwt = float(np.mean(diffs))
        
    return {'ACC': acc, 'BWT': bwt}
