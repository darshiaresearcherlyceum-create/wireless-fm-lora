import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

def compute_rzf_teacher_precoder(channels: torch.Tensor, alpha: float = 0.018) -> torch.Tensor:
    """
    Computes separate analytical Regularized Zero-Forcing (RZF) teacher precoder
    using numerically stable linear algebra solve: W_RZF = H^H (H H^H + alpha I)^(-1).
    channels: [B, N_sub, 1024]
    Returns: [B, N_sub, 1024] normalized teacher precoder W_RZF
    """
    B_size, N_sub = channels.shape[0], channels.shape[1]
    K, N_BS = 8, 64
    ch_real = channels[:, :, :512].reshape(B_size, N_sub, K, N_BS)
    ch_imag = channels[:, :, 512:].reshape(B_size, N_sub, K, N_BS)
    H_c = torch.complex(ch_real, ch_imag) # [B, N_sub, K, N_BS]
    
    # Gram matrix HH: [B, N_sub, K, K]
    HH = torch.matmul(H_c, H_c.mH)
    eye_K = torch.eye(K, dtype=H_c.dtype, device=H_c.device).unsqueeze(0).unsqueeze(0)
    reg_mat = HH + alpha * eye_K
    
    # Solve (HH + alpha*I) X = H => X = (HH + alpha*I)^(-1) H  [B, N_sub, K, N_BS]
    # W_rzf = X^H = H^H (HH + alpha*I)^(-1)  [B, N_sub, N_BS, K]
    X = torch.linalg.solve(reg_mat, H_c)
    W_rzf = X.mH # [B, N_sub, N_BS, K]
    
    # Transmit power normalization per column: ||w_k||_2 = 1 => total power = K = 8
    w_norm = torch.linalg.norm(W_rzf, dim=2, keepdim=True) + 1e-8
    W_norm = W_rzf / w_norm
    
    return torch.cat([
        W_norm.real.reshape(B_size, N_sub, 512),
        W_norm.imag.reshape(B_size, N_sub, 512)
    ], dim=-1)

def compute_detection_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_bit: float = 1.0
) -> torch.Tensor:
    """
    Computes joint Symbol MSE + Bit BCEWithLogits loss for MIMO Detection (Task 1).
    QPSK symbol-to-bit mapping:
      bit 0 -> +1/sqrt(2)
      bit 1 -> -1/sqrt(2)
    Logit for bit 1: -pred * sqrt(2)
    """
    loss_sym = F.mse_loss(pred, target)
    target_bits = (target < 0.0).float()
    pred_logits = -pred * (2.0 ** 0.5)
    loss_bit = F.binary_cross_entropy_with_logits(pred_logits, target_bits)
    return loss_sym + lambda_bit * loss_bit

def compute_precoding_loss(
    pred: torch.Tensor,
    true_channels: torch.Tensor,
    snr_db: float = 10.0,
    lambda_rzf: float = 1.0,
    lambda_rate: float = 0.02
) -> torch.Tensor:
    """
    Computes multi-objective Precoding Loss (Task 2) combining:
      1. Normalized Frobenius MSE loss against analytical RZF teacher target
      2. Differentiable sum-rate loss (-R_sum)
    """
    with torch.no_grad():
        w_rzf = compute_rzf_teacher_precoder(true_channels, alpha=0.018)
        
    num = torch.sum((pred - w_rzf) ** 2, dim=-1)
    denom = torch.sum(w_rzf ** 2, dim=-1) + 1e-8
    loss_rzf = (num / denom).mean()
    
    loss_rate = compute_differentiable_sum_rate_loss(pred, true_channels, snr_db=snr_db)
    return lambda_rzf * loss_rzf + lambda_rate * loss_rate

def compute_differentiable_sum_rate_loss(
    preds: torch.Tensor,
    true_channels: torch.Tensor,
    snr_db: float = 10.0
) -> torch.Tensor:
    """
    Computes negative differentiable achievable sum-rate loss for MU-MIMO precoding.
    preds: [B, N_sub, 1024] - predicted beamforming matrix W (real, imag)
    true_channels: [B, N_sub, 1024] - channel matrix H (real, imag)
    snr_db: Signal-to-Noise Ratio in dB
    """
    B_size = preds.shape[0]
    N_sub, K, N_BS = 64, 8, 64
    
    # Reconstruct complex channel matrix H: [B, N_sub, K, N_BS]
    ch_real = true_channels[:, :, :512].reshape(B_size, N_sub, K, N_BS)
    ch_imag = true_channels[:, :, 512:].reshape(B_size, N_sub, K, N_BS)
    H_mat = torch.complex(ch_real, ch_imag)
    
    # Reconstruct complex precoding matrix W: [B, N_sub, N_BS, K]
    prec_real = preds[:, :, :512].reshape(B_size, N_sub, N_BS, K)
    prec_imag = preds[:, :, 512:].reshape(B_size, N_sub, N_BS, K)
    W_mat = torch.complex(prec_real, prec_imag)
    
    # Power normalization per BS antenna column (dim=2: N_BS)
    # W_norm: [B, N_sub, 1, K]
    w_sq = (prec_real ** 2 + prec_imag ** 2).sum(dim=2, keepdim=True)
    w_norm = torch.sqrt(w_sq + 1e-8)
    W_mat_norm = W_mat / w_norm
    
    # Effective Multi-User Channel: [B, N_sub, K, K]
    HW = torch.matmul(H_mat, W_mat_norm)
    HW_sq = (HW.real ** 2) + (HW.imag ** 2) # [B, N_sub, K, K]
    
    # Desired signal power along diagonal: [B, N_sub, K]
    signal = torch.diagonal(HW_sq, dim1=2, dim2=3)
    
    # Total received power and multi-user interference
    total_power = torch.sum(HW_sq, dim=-1) # [B, N_sub, K]
    interference = total_power - signal    # [B, N_sub, K]
    
    noise_power = 10.0 ** (-snr_db / 10.0)
    
    sinr = signal / (interference + noise_power + 1e-8)
    user_rates = torch.log2(1.0 + sinr) # [B, N_sub, K]
    sum_rate = user_rates.sum(dim=-1).mean() # average sum-rate per subcarrier
    
    # Negative sum-rate to maximize communication rate during gradient descent
    return -sum_rate

def compute_task_loss(
    task_id: int, 
    pred: torch.Tensor, 
    target: torch.Tensor,
    inputs: Optional[torch.Tensor] = None,
    true_channels: Optional[torch.Tensor] = None,
    snr: Optional[torch.Tensor] = None,
    lambda_prec_mse: float = 1.0,
    lambda_prec_rate: float = 0.02
) -> torch.Tensor:
    """
    Computes appropriate loss function per wireless task:
      Task 0 (CE): Direct NMSE Loss on Channel Estimation
      Task 1 (MIMO DET): Joint Symbol MSE + Bit BCEWithLogits
      Task 2 (PREC): Multi-objective Loss = lambda_rzf * Normalized_Frobenius_Loss(W_hat, W_RZF) + lambda_rate * (-SumRate)
      Task 3 (DEC): Binary Cross Entropy with Logits
      Task 4 (LOC): Smooth L1 + Anti-Collapse Variance Regularizer
    """
    if task_id == 0:
        # Normalized Mean Square Error (NMSE) Loss for final_CE with SNR-Aware Balancing and Explicit LS Improvement
        num = torch.sum((pred - target) ** 2, dim=-1) # [B, N_sub]
        denom = torch.sum(target ** 2, dim=-1) + 1e-8 # [B, N_sub]
        sample_nmse = (num / denom).mean(dim=-1)       # [B]
        
        # Explicitly optimize to beat the Least Squares (LS) channel estimate
        if inputs is not None:
            ls_ce = inputs[:, :, :1024] # Raw Least Squares baseline (Y_p)
            num_ls = torch.sum((ls_ce - target) ** 2, dim=-1)
            sample_nmse_ls = (num_ls / denom).mean(dim=-1) # [B]
            
            # Relative ratio term: rewards final_CE for achieving NMSE < LS_NMSE across all SNRs
            rel_ratio = sample_nmse / (sample_nmse_ls + 1e-8)
            # Hinge penalty: provides direct corrective gradient if final_CE fails to improve over LS
            penalty_worse_than_ls = F.relu(sample_nmse - sample_nmse_ls)
            
            ce_metric_loss = sample_nmse + 0.5 * rel_ratio + 1.0 * penalty_worse_than_ls
        else:
            ce_metric_loss = sample_nmse
        
        if snr is not None:
            snr_vals = snr.view(-1).float()            # [B]
            # SNR-aware balancing across [-5, 0, 5, 10, 15, 20, 25] dB:
            # Low SNR (-5 dB) error is ~0.9, high SNR (25 dB) error is ~0.015.
            # Compensates for linear NMSE scale using smooth square-root SNR weighting 10^((SNR - 10)/20)
            # Normalized by batch mean weight to strictly preserve canonical loss magnitude.
            snr_weights = torch.pow(10.0, (snr_vals - 10.0) / 20.0)
            snr_weights = snr_weights / (snr_weights.mean().detach() + 1e-8)
            loss_ce = 10.0 * (ce_metric_loss * snr_weights).mean()
        else:
            loss_ce = 10.0 * ce_metric_loss.mean()
            
        return loss_ce
        
    elif task_id == 1:
        return compute_detection_loss(pred, target, lambda_bit=1.0)
        
    elif task_id == 2:
        if true_channels is not None:
            ch_for_rzf = true_channels
        elif inputs is not None:
            ch_for_rzf = inputs[:, :, :1024]
        else:
            ch_for_rzf = target
        snr_val = float(snr.mean().item()) if snr is not None else 10.0
        return compute_precoding_loss(
            pred, 
            ch_for_rzf, 
            snr_db=snr_val, 
            lambda_rzf=lambda_prec_mse, 
            lambda_rate=lambda_prec_rate
        )
        
    elif task_id == 3:
        return F.binary_cross_entropy_with_logits(pred, target)
        
    elif task_id == 4:
        # User 3D Localization: Smooth L1 Loss + Anti-Collapse Variance Regularizer
        loss_smooth = F.smooth_l1_loss(pred, target, beta=0.01)
        pred_users = pred.view(-1, 8, 3)
        var_xy = torch.var(pred_users[:, :, :2], dim=(0, 1), unbiased=False).mean()
        var_penalty = torch.clamp(0.05 - var_xy, min=0.0)
        return 50.0 * loss_smooth + 5.0 * var_penalty
        
    else:
        return F.mse_loss(pred, target)

def compute_rank_complexity_penalty(w: torch.Tensor, ranks: list, beta: float = 0.005) -> torch.Tensor:
    """
    Penalizes higher LoRA rank selection using soft expectation:
      Penalty = beta * sum_r (w_r * r)
    """
    device = w.device
    ranks_tensor = torch.tensor(ranks, dtype=torch.float32, device=device)
    expected_rank = (w * ranks_tensor).sum(dim=-1) # [B]
    return beta * expected_rank.mean()
