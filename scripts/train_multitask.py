"""
scripts/train_multitask.py
Joint multi-task training on a single environment or seen environments.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import torch
import torch.optim as optim
from src.utils.seed import set_seed
from src.utils.config import load_config, save_config
from src.utils.logger import ExperimentLogger
from src.models.wireless_fm import EnvironmentAwareWirelessFM
from src.data.dataset import get_dataloaders
from src.train.trainer import train_one_epoch_multitask
from src.eval.evaluator import evaluate_all_tasks

def main():
    parser = argparse.ArgumentParser(description="Multi-Task Training on Wireless Environments")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--env", type=str, default="Urban_Macro")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit sample count for fast CPU runs")
    parser.add_argument("--exp_name", type=str, default="proposed")
    parser.add_argument("--log_dir", type=str, default="logs/day11")
    parser.add_argument("--file_suffix", type=str, default="", help="Suffix for output filenames, e.g. '(2)'")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    set_seed(cfg['system']['seed'])
    device = cfg['system']['device'] if torch.cuda.is_available() else "cpu"
    
    exp_name = args.exp_name
    log_dir = args.log_dir or cfg['system']['log_dir']
    
    suffix = args.file_suffix
    log_filename = f"run_log{suffix}.txt" if suffix else "run_log.txt"
    csv_filename = f"metrics{suffix}.csv" if suffix else "metrics.csv"
    snr_filename = f"snr_evaluation{suffix}.csv" if suffix else "snr_evaluation.csv"
    
    logger = ExperimentLogger(
        log_dir=log_dir,
        exp_name=exp_name,
        log_filename=log_filename,
        csv_filename=csv_filename
    )
    logger.log(f"Starting Multi-Task Training on environment: {args.env} (Device: {device})")
    
    # Save config snapshot with suffix so existing config is not overwritten
    cfg_target = os.path.join(logger.run_dir, f"config{suffix}.yaml" if suffix else "config.yaml")
    save_config(cfg, cfg_target)
    
    # Load Dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        env_name=args.env,
        data_dir=cfg['data']['data_dir'],
        batch_size=cfg['training']['batch_size'],
        train_split=cfg['data']['train_split'],
        val_split=cfg['data']['val_split'],
        max_samples=args.max_samples,
        seed=cfg['system']['seed']
    )
    
    # Initialize Proposed Model
    model = EnvironmentAwareWirelessFM(
        ranks=cfg['model']['rank_set'],
        feature_dim=cfg['wireless']['feature_dim'],
        semantic_coupling=cfg['model']['semantic_coupling'],
        lora_alpha=cfg['model'].get('lora_alpha', 1.0)
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = model.get_trainable_parameters()
    num_trainable_params = sum(p.numel() for p in trainable_params)
    frozen_params = total_params - num_trainable_params
    logger.log("--- Model Parameter Counts ---")
    logger.log(f"  Total parameters:     {total_params:,}")
    logger.log(f"  Trainable parameters: {num_trainable_params:,}")
    logger.log(f"  Frozen parameters:    {frozen_params:,}")
    optimizer = optim.AdamW(
        trainable_params,
        lr=float(cfg['training']['learning_rate']),
        weight_decay=float(cfg['training']['weight_decay'])
    )
    
    epochs = args.epochs or cfg['training'].get('epochs_per_env', 10)
    warmup_epochs = cfg['training'].get('warmup_epochs', 0)
    
    # Schedulers: CosineAnnealingLR matching Static LoRA
    if warmup_epochs > 0 and epochs > warmup_epochs:
        warmup_sched = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-5)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        
    tau = cfg['training']['gumbel_tau_init']
    
    best_val_loss = float("inf")
    ckpt_dir = os.path.join(cfg['system']['save_dir'], "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, f"{exp_name}{suffix}_best.pt" if suffix else f"{exp_name}_best.pt")
    
    # Initial gradient verification across modules
    logger.log("\n--- Verifying Gradient Flow Across Trainable Modules ---")
    model.train()
    sample_batch = next(iter(train_loader))
    sample_scene = sample_batch['scene'].to(device)
    B_s = sample_scene.shape[0]
    
    with torch.no_grad():
        cls_emb = model.scene_encoder(sample_scene)
    z_s = model.scene_proj(cls_emb)
    
    optimizer.zero_grad()
    from src.train.losses import compute_task_loss, compute_rank_complexity_penalty
    sample_loss = 0.0
    for t in range(5):
        s_in = sample_batch['inputs'][:, t].to(device)
        s_tgt = sample_batch[f'target_{t}'].to(device)
        s_tids = torch.full((B_s,), t, dtype=torch.long, device=device)
        s_out = model(s_in, sample_scene, s_tids, z_s=z_s)
        t_loss = compute_task_loss(t, s_out[f'out_{t}'], s_tgt, inputs=s_in)
        r_pen = compute_rank_complexity_penalty(s_out['w'], model.ranks)
        sample_loss = sample_loss + (t_loss / 5.0) + (r_pen / 5.0)
    sample_loss.backward()
    
    modules_to_check = {
        'hypernetwork': model.hypernetwork,
        'rank_gate': model.rank_gate,
        'task_heads': model.task_heads,
        'scene_proj': model.scene_proj,
        'input_proj': model.input_proj
    }
    for m_name, m_mod in modules_to_check.items():
        grads = [p.grad for p in m_mod.parameters() if p.requires_grad and p.grad is not None]
        non_zero = any(g.abs().sum() > 0 for g in grads)
        is_finite = all(torch.all(torch.isfinite(g)) for g in grads)
        logger.log(f"  Module [{m_name:12s}] -> Gradients non-zero: {non_zero} | Finite: {is_finite}")
        assert non_zero and is_finite, f"Gradient verification failed for module {m_name}"
        
    optimizer.zero_grad()
    logger.log("--- Gradient Flow Verification Passed ---\n")
    
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_metrics = train_one_epoch_multitask(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            tau=tau,
            beta=cfg['training']['beta_rank_penalty'],
            device=device
        )
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Evaluate on validation set
        val_eval = evaluate_all_tasks(model, val_loader, snr_db=10.0, device=device)
        epoch_runtime_s = time.time() - epoch_start_time
        
        # Format rank distribution string
        rank_dist_str = ", ".join([f"r{r}:{train_metrics.get(f'pct_rank_{r}', 0.0):.1f}%" for r in model.ranks])
        
        logger.log(
            f"Epoch {epoch:02d}/{epochs:02d} (lr={current_lr:.2e}) | "
            f"Train Loss: {train_metrics['loss']:.4f} (CE:{train_metrics['loss_ce']:.4f}, DET:{train_metrics['loss_det']:.4f}, "
            f"PREC:{train_metrics['loss_prec']:.4f}, DEC:{train_metrics['loss_dec']:.4f}, LOC:{train_metrics['loss_loc']:.4f}) | "
            f"Avg Rank: {train_metrics['avg_selected_rank']:.1f} [{rank_dist_str}] | "
            f"Val -> CE NMSE: {val_eval['CE_NMSE_dB']:.2f}dB | DET BER: {val_eval['DET_BER']:.4f} | "
            f"PREC: {val_eval['PREC_SumRate']:.2f}b/s/Hz | DEC BER: {val_eval['DEC_BER']:.4f} | "
            f"LOC RMSE: {val_eval['LOC_RMSE_m']:.2f}m | Runtime: {epoch_runtime_s:.1f}s"
        )
        
        # Anneal Gumbel temperature
        tau = max(cfg['training']['gumbel_tau_min'], tau * cfg['training']['gumbel_tau_decay'])
        
        # Save metrics with only canonical columns
        logged_dict = {
            "loss": round(train_metrics["loss"], 4),
            "CE_NMSE_dB": round(val_eval["CE_NMSE_dB"], 2),
            "DET_BER": round(val_eval["DET_BER"], 4),
            "PREC_SumRate": round(val_eval["PREC_SumRate"], 2),
            "DEC_BER": round(val_eval["DEC_BER"], 4),
            "LOC_RMSE_m": round(val_eval["LOC_RMSE_m"], 2),
            "trainable_params": num_trainable_params,
            "runtime_s": round(epoch_runtime_s, 2)
        }
        logger.log_metrics(epoch, logged_dict)
        
        if train_metrics['loss'] < best_val_loss:
            best_val_loss = train_metrics['loss']
            torch.save(model.state_dict(), best_ckpt_path)
            
    logger.log(f"Training completed. Best model checkpoint saved to: {best_ckpt_path}")
    logger.close()
    
    # Run full SNR grid evaluation on best checkpoint and save snr_evaluation.csv
    from scripts.run_day11_evaluation import evaluate_method_across_snr
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device, weights_only=False))
    model.eval()
    evaluate_method_across_snr(
        method_name="proposed",
        model=model,
        test_loader=test_loader,
        output_dir=logger.run_dir,
        device=device,
        snr_csv_filename=snr_filename,
        log_filename=log_filename
    )
    
    # Also ensure runlog(2).txt is available alongside run_log(2).txt if suffix is '(2)'
    if suffix == "(2)":
        alt_log = os.path.join(logger.run_dir, "runlog(2).txt")
        if os.path.exists(logger.log_file):
            import shutil
            shutil.copy(logger.log_file, alt_log)
    
    # Copy metrics to results/metrics directory
    metrics_dir = os.path.join(cfg['system']['save_dir'], "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    if os.path.exists(logger.csv_file):
        import shutil
        shutil.copy(logger.csv_file, os.path.join(metrics_dir, f"{exp_name}{suffix}_metrics.csv" if suffix else f"{exp_name}_metrics.csv"))

if __name__ == "__main__":
    main()
