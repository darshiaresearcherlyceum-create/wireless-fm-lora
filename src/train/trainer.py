import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List
import copy
import numpy as np

from tqdm import tqdm
from src.train.losses import compute_task_loss, compute_rank_complexity_penalty
from src.train.ewc import compute_ewc_loss

def train_one_epoch_multitask(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    tau: float = 1.0,
    beta: float = 0.005,
    task_weights: Optional[List[float]] = None,
    lambda_prec_mse: float = 1.0,
    lambda_prec_rate: float = 0.02,
    ewc_lambda: float = 0.0,
    prev_params: Optional[Dict[str, torch.Tensor]] = None,
    fisher_diags: Optional[Dict[str, torch.Tensor]] = None,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Performs joint multi-task training across all 5 physical layer tasks
    using memory-efficient per-task backward accumulation with task loss balancing.
    
    Task Weights Rationale:
      - Task 0 (CE, 2.0): Direct residual refinement over LS observation.
      - Task 1 (DET, 1.0): Reliable constellation classification.
      - Task 2 (PREC, 2.0): Direct multi-objective precoding with differentiable sum-rate term.
      - Task 3 (DEC, 1.0): Stable binary cross entropy bit decoding.
      - Task 4 (LOC, 3.0): Prioritizes spatial localization to reach sub-20m accuracy.
    """
    if task_weights is None:
        task_weights = [1.0, 1.0, 1.0, 1.0, 1.0]  # Uniform task weights matching Static LoRA baseline
        
    model.train()
    total_loss = 0.0
    total_task_loss = 0.0
    total_rank_pen = 0.0
    total_ewc_loss = 0.0
    task_losses = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    
    num_ranks = len(model.ranks)
    rank_counts = np.zeros(num_ranks, dtype=np.float64)
    total_rank_samples = 0
    ranks_arr = np.array(model.ranks, dtype=np.float64)
    
    num_samples = len(dataloader.dataset)
    
    pbar = tqdm(dataloader, desc="Training Multi-Task", leave=False)
    for batch in pbar:
        scene = batch['scene'].to(device)
        B_size = scene.shape[0]
        optimizer.zero_grad()
        
        # Precompute frozen scene embedding and scene projection once per batch
        with torch.no_grad():
            cls_emb = model.scene_encoder(scene)
        z_s = model.scene_proj(cls_emb)
            
        step_loss = 0.0
        for t in range(5):
            inputs = batch['inputs'][:, t].to(device)
            target = batch[f'target_{t}'].to(device)
            snr_tensor = batch.get('snr')
            if snr_tensor is not None:
                snr_tensor = snr_tensor.to(device)
                
            task_ids = torch.full((B_size,), t, dtype=torch.long, device=device)
            
            outputs = model(inputs, scene, task_ids, tau=tau, z_s=z_s, snr=snr_tensor)
            pred = outputs[f'out_{t}']
            
            true_channels = batch['target_0'].to(device) if 'target_0' in batch else None
            task_loss = compute_task_loss(
                t, 
                pred, 
                target, 
                inputs=inputs, 
                true_channels=true_channels,
                snr=snr_tensor,
                lambda_prec_mse=lambda_prec_mse,
                lambda_prec_rate=lambda_prec_rate
            )
            rank_pen = compute_rank_complexity_penalty(outputs['w'], model.ranks, beta=beta)
            
            w_t = task_weights[t]
            sub_loss = (w_t * task_loss / 5.0) + (rank_pen / 5.0)
            step_loss = step_loss + sub_loss
            
            task_loss_val = task_loss.item()
            task_losses[t] += task_loss_val * B_size
            total_task_loss += (w_t * task_loss_val / 5.0) * B_size
            total_rank_pen += (rank_pen.item() / 5.0) * B_size
            
            # Record rank selection distribution
            with torch.no_grad():
                w_detached = outputs['w'].detach().cpu().numpy() # [B, num_ranks]
                sel_idx = np.argmax(w_detached, axis=-1)
                for idx in sel_idx:
                    rank_counts[idx] += 1
                total_rank_samples += B_size
            
        # Optional EWC penalty
        if prev_params is not None and fisher_diags is not None and ewc_lambda > 0:
            ewc_val = compute_ewc_loss(model, prev_params, fisher_diags)
            step_loss = step_loss + (ewc_lambda * ewc_val)
            total_ewc_loss += (ewc_lambda * ewc_val.item()) * B_size
            
        step_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.get_trainable_parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += step_loss.item() * B_size
        pbar.set_postfix({'batch_loss': f"{step_loss.item():.4f}"})
        
    rank_pcts = (rank_counts / max(total_rank_samples, 1)) * 100.0
    avg_rank = np.sum(rank_pcts / 100.0 * ranks_arr)
    
    metrics_res = {
        'loss': total_loss / num_samples,
        'task_loss': total_task_loss / num_samples,
        'loss_ce': task_losses[0] / num_samples,
        'loss_det': task_losses[1] / num_samples,
        'loss_prec': task_losses[2] / num_samples,
        'loss_dec': task_losses[3] / num_samples,
        'loss_loc': task_losses[4] / num_samples,
        'rank_penalty': total_rank_pen / num_samples,
        'ewc_loss': total_ewc_loss / num_samples,
        'avg_selected_rank': float(avg_rank)
    }
    for i, r_val in enumerate(model.ranks):
        metrics_res[f'pct_rank_{r_val}'] = float(rank_pcts[i])
        
    return metrics_res
