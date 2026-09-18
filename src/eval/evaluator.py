import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from src.eval.metrics import (
    evaluate_channel_estimation_nmse,
    evaluate_mimo_detection_nmse,
    evaluate_mimo_detection_ber,
    evaluate_precoding_sum_rate,
    evaluate_channel_decoding,
    evaluate_user_localization_error_distance,
    evaluate_user_localization_rmse
)

def evaluate_all_tasks(
    model: nn.Module,
    dataloader: DataLoader,
    snr_db: float = 10.0,
    device: str = "cpu",
    baseline_model: Optional[nn.Module] = None
) -> Dict[str, float]:
    """
    Evaluates model across all 5 physical layer wireless tasks.
    Returns:
      5 Canonical Wireless Metrics:
        - CE_NMSE_dB: Channel Estimation NMSE (dB) ↓
        - DET_BER: MIMO Detection Bit Error Rate (BER) ↓
        - PREC_SumRate: Precoding Achievable Sum-Rate (bits/s/Hz) ↑
        - DEC_BER: Channel Decoding Bit Error Rate (BER) ↓
        - DEC_BLER: Channel Decoding Block Error Rate (BLER) ↓
        - LOC_RMSE_m: User Localization RMSE (meters) ↓
      Diagnostic Metrics:
        - DET_SER: MIMO Detection Symbol Error Rate ↓
        - DET_NMSE_dB: MIMO Detection NMSE (dB) ↓
        - LOC_Error_Distance_m: User Localization Mean Error Distance (m) ↓
    """
    eval_model = model if baseline_model is None else baseline_model
    eval_model.eval()
    
    if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, 'set_snr'):
        dataloader.dataset.set_snr(snr_db)
    
    preds_all = {t: [] for t in range(5)}
    targets_all = {t: [] for t in range(5)}
    
    with torch.no_grad():
        for batch in dataloader:
            scene = batch['scene'].to(device)
            if baseline_model is None:
                cls_emb = eval_model.scene_encoder(scene)
                z_s = eval_model.scene_proj(cls_emb)
            else:
                z_s = None
                
            for t in range(5):
                inputs = batch['inputs'][:, t].to(device)
                target = batch[f'target_{t}'].to(device)
                task_ids = torch.full((inputs.shape[0],), t, dtype=torch.long, device=device)
                
                if baseline_model is None:
                    outputs = eval_model(inputs, scene, task_ids, hard=True, z_s=z_s)
                else:
                    outputs = eval_model(inputs, task_ids)
                    
                preds_all[t].append(outputs[f'out_{t}'].cpu().numpy())
                targets_all[t].append(target.cpu().numpy())
                
    for t in range(5):
        preds_all[t] = np.concatenate(preds_all[t], axis=0)
        targets_all[t] = np.concatenate(targets_all[t], axis=0)
        
    nmse_ce = evaluate_channel_estimation_nmse(preds_all[0], targets_all[0])
    nmse_det = evaluate_mimo_detection_nmse(preds_all[1], targets_all[1])
    ber_det, ser_det = evaluate_mimo_detection_ber(preds_all[1], targets_all[1])
    sum_rate = evaluate_precoding_sum_rate(preds_all[2], targets_all[0], snr_db=snr_db)
    
    # Compute separate classical baseline rates (RZF and ZF)
    from src.eval.metrics import compute_classical_rzf_precoder, compute_classical_zf_precoder
    w_rzf = compute_classical_rzf_precoder(targets_all[0])
    sum_rate_rzf = evaluate_precoding_sum_rate(w_rzf, targets_all[0], snr_db=snr_db)
    w_zf = compute_classical_zf_precoder(targets_all[0])
    sum_rate_zf = evaluate_precoding_sum_rate(w_zf, targets_all[0], snr_db=snr_db)
    
    ber_dec, bler_dec = evaluate_channel_decoding(preds_all[3], targets_all[3])
    loc_error_dist = evaluate_user_localization_error_distance(preds_all[4], targets_all[4])
    rmse_loc = evaluate_user_localization_rmse(preds_all[4], targets_all[4])
    
    return {
        'CE_NMSE_dB': nmse_ce,
        'DET_BER': ber_det,
        'PREC_SumRate': sum_rate,
        'PREC_SumRate_Proposed': sum_rate,
        'PREC_SumRate_RZF': sum_rate_rzf,
        'PREC_SumRate_ZF': sum_rate_zf,
        'DEC_BER': ber_dec,
        'LOC_RMSE_m': rmse_loc,
        'DET_SER': ser_det,
        'DET_NMSE_dB': nmse_det,
        'LOC_Error_Distance_m': loc_error_dist
    }
