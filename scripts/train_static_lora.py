import os
import sys
import csv
import time
import datetime
import argparse
import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.seed import set_seed
from src.utils.config import load_config
from src.data.dataset import get_dataloaders
from src.models.baselines import StaticLoRAModel
from src.train.losses import compute_task_loss
from src.eval.evaluator import evaluate_all_tasks
from scripts.run_day11_evaluation import evaluate_method_across_snr

def log_msg(msg: str, log_file: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"{timestamp} [INFO] {msg}"
    print(formatted, flush=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")

def train_static_lora(
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
    device,
    log_file,
    best_ckpt_path
):
    """
    Train Static LoRA baseline on all 5 wireless tasks.
    Fixed LoRA rank = 8, no hypernetwork.
    """
    model.train()
    optimizer = optim.AdamW(
        model.get_trainable_parameters(),
        lr=lr,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = []
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            batch_inputs = batch["inputs"].to(device)
            batch_size = batch_inputs.shape[0]
            step_loss = 0.0

            # Train all 5 tasks
            for t in range(5):
                inputs = batch["inputs"][:, t].to(device)
                target = batch[f"target_{t}"].to(device)
                task_ids = torch.full((batch_size,), t, dtype=torch.long, device=device)
                snr_t = batch.get('snr')
                if snr_t is not None:
                    snr_t = snr_t.to(device)
                    
                outputs = model(inputs, task_ids, snr=snr_t)
                pred = outputs[f"out_{t}"]
                true_ch = batch['target_0'].to(device) if 'target_0' in batch else None
                
                loss = compute_task_loss(
                    t,
                    pred,
                    target,
                    inputs=inputs,
                    true_channels=true_ch,
                    snr=snr_t
                )
                step_loss = step_loss + (loss / 5.0)

            step_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.get_trainable_parameters(),
                max_norm=1.0
            )
            optimizer.step()
            total_loss += float(step_loss.item())
            num_batches += 1

        scheduler.step()
        epoch_loss = total_loss / max(num_batches, 1)
        runtime = time.time() - epoch_start

        # Evaluate on validation set
        val_eval = evaluate_all_tasks(
            model=model,
            dataloader=val_loader,
            snr_db=10.0,
            device=device,
            baseline_model=model
        )

        log_msg(
            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_loss:.4f} | "
            f"Val -> CE NMSE: {val_eval['CE_NMSE_dB']:.2f} dB | DET BER: {val_eval['DET_BER']:.4f} | "
            f"PREC: {val_eval['PREC_SumRate']:.2f} bps/Hz | DEC BER: {val_eval['DEC_BER']:.4f} | "
            f"LOC RMSE: {val_eval['LOC_RMSE_m']:.2f} m | Runtime: {runtime:.1f}s",
            log_file
        )

        history.append({
            "epoch": epoch,
            "loss": epoch_loss,
            "CE_NMSE_dB": val_eval["CE_NMSE_dB"],
            "DET_BER": val_eval["DET_BER"],
            "PREC_SumRate": val_eval["PREC_SumRate"],
            "DEC_BER": val_eval["DEC_BER"],
            "LOC_RMSE_m": val_eval["LOC_RMSE_m"],
            "runtime_s": runtime
        })

        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            torch.save(model.state_dict(), best_ckpt_path)

    return history

def main():
    parser = argparse.ArgumentParser(description="Train Day 11 Static LoRA Baseline")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--env", type=str, default="Urban_Macro")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--log_dir", type=str, default="logs/day11/static_lora")
    parser.add_argument("--save_path", type=str, default="results/checkpoints/static_lora_day11_best.pt")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    log_file = os.path.join(args.log_dir, "run_log.txt")
    metrics_csv = os.path.join(args.log_dir, "metrics.csv")

    cfg = load_config(args.config)
    set_seed(cfg["system"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log_msg(f"=== Starting Multi-Task Training: Static LoRA (r=8) on {args.env} ===", log_file)
    log_msg(f"Device: {device} | Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}", log_file)

    train_loader, val_loader, test_loader = get_dataloaders(
        env_name=args.env,
        data_dir=cfg["data"]["data_dir"],
        batch_size=args.batch_size,
        train_split=cfg["data"]["train_split"],
        val_split=cfg["data"]["val_split"],
        seed=cfg["system"]["seed"]
    )

    model = StaticLoRAModel(rank=8).to(device)

    trainable_params = sum(p.numel() for p in model.get_trainable_parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = total_params - trainable_params
    log_msg("--- Model Parameter Counts ---", log_file)
    log_msg(f"  Total parameters:     {total_params:,}", log_file)
    log_msg(f"  Trainable parameters: {trainable_params:,}", log_file)
    log_msg(f"  Frozen parameters:    {frozen_params:,}", log_file)
    log_msg(f"  LoRA Rank:            r=8 (Fixed, No Hypernetwork)", log_file)

    history = train_static_lora(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        log_file=log_file,
        best_ckpt_path=args.save_path
    )

    log_msg(f"Training completed. Best model checkpoint saved to: {args.save_path}", log_file)

    # Save training metrics.csv
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["epoch", "loss", "CE_NMSE_dB", "DET_BER", "PREC_SumRate", "DEC_BER", "LOC_RMSE_m", "trainable_params", "runtime_s"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({
                "epoch": row["epoch"],
                "loss": round(row["loss"], 4),
                "CE_NMSE_dB": round(row["CE_NMSE_dB"], 2),
                "DET_BER": round(row["DET_BER"], 4),
                "PREC_SumRate": round(row["PREC_SumRate"], 2),
                "DEC_BER": round(row["DEC_BER"], 4),
                "LOC_RMSE_m": round(row["LOC_RMSE_m"], 2),
                "trainable_params": trainable_params,
                "runtime_s": round(row["runtime_s"], 2)
            })
    log_msg(f"Saved training progression to: {metrics_csv}", log_file)

    # Run complete SNR Grid Evaluation on held-out test split
    log_msg("=== Commencing Test Split Evaluation across SNR Grid ===", log_file)
    model.load_state_dict(torch.load(args.save_path, map_location=device, weights_only=False))
    model.eval()

    evaluate_method_across_snr(
        method_name="static_lora",
        model=model,
        test_loader=test_loader,
        output_dir=args.log_dir,
        device=device
    )

if __name__ == "__main__":
    main()