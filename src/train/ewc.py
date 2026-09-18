import torch
import torch.nn as nn
from typing import Dict, Optional
from torch.utils.data import DataLoader
from src.train.losses import compute_task_loss

def compute_fisher_diagonal(
    model: nn.Module,
    dataloader: DataLoader,
    num_samples: int = 300,
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Computes diagonal Fisher Information Matrix (FIM) for Elastic Weight Consolidation (EWC)
    over trainable hypernetwork and task head parameters.
    """
    model.eval()
    fisher = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            fisher[name] = torch.zeros_like(p.data)
            
    count = 0
    model.zero_grad()
    
    for batch in dataloader:
        scene = batch['scene'].to(device)
        B_size = scene.shape[0]
        
        for t in range(5):
            inputs = batch['inputs'][:, t].to(device)
            target = batch[f'target_{t}'].to(device)
            task_ids = torch.full((B_size,), t, dtype=torch.long, device=device)
            
            outputs = model(inputs, scene, task_ids)
            pred = outputs[f'out_{t}']
            loss = compute_task_loss(t, pred, target)
            
            model.zero_grad()
            loss.backward()
            
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[name] += (p.grad.data ** 2) * B_size
                    
        count += B_size
        if count >= num_samples:
            break
            
    for name in fisher:
        fisher[name] = fisher[name] / max(count, 1)
        fisher[name] = torch.clamp(fisher[name], min=1e-8)
        
    model.zero_grad()
    return fisher

def compute_ewc_loss(
    model: nn.Module,
    prev_params: Dict[str, torch.Tensor],
    fisher_diags: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    Computes EWC regularizer: 0.5 * sum_i F_i * (theta_i - theta_i^*)^2
    """
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, p in model.named_parameters():
        if p.requires_grad and name in prev_params and name in fisher_diags:
            f_diag = fisher_diags[name].to(p.device)
            p_old = prev_params[name].to(p.device)
            loss = loss + (f_diag * (p - p_old) ** 2).sum()
    return 0.5 * loss
