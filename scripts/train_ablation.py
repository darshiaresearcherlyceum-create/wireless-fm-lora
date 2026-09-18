"""
scripts/train_ablation.py
Day 13 Ablation Training and Evaluation Suite:
  A1: No scene encoder (ViT removed, zero-filled scene embedding)
  A2: No adaptive rank (Force rank = 8, no Gumbel rank gate)
  A3: No hypernetwork (Static LoRA r=8 with same task heads and training recipe)
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import datetime
import argparse
import csv
import numpy as np
import torch
import torch.optim as optim

from src.utils.seed import set_seed
from src.utils.config import load_config, save_config
from src.data.dataset import get_dataloaders
from src.train.losses import compute_task_loss
from src.eval.metrics import (
    evaluate_channel_estimation_nmse,
    evaluate_mimo_detection_ber,
    evaluate_precoding_sum_rate,
    evaluate_channel_decoding,
    evaluate_user_localization_rmse
)
from src.models.ablations import (
    AblationA1_NoScene,
    AblationA2_NoAdaptiveRank,
    AblationA3_NoHypernet
)


def log_msg(msg: str, log_file: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"{timestamp} [INFO] {msg}"
    print(formatted, flush=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")


def evaluate_ablation_checkpoint_snr(
    model,
    test_loader,
    output_dir: str,
    device: str,
    is_a3: bool = False,
    snr_grid=[-5, 0, 5, 10, 15, 20, 25],
    log_file: str = ""
):
    """
    Evaluates saved checkpoint on the unseen test split across SNR grid.
    Strictly follows:
      - Rule 1: No RZF inside model.
      - Rule 2: Detection metric labeled DET_BER_MMSE_init.
      - Rule 3: LOC RMSE evaluated on test split.
      - Rule 4: Not copying epoch metrics into SNR table.
      - Rule 5: No buzzwords.
    """
    snr_csv_path = os.path.join(output_dir, "snr_evaluation.csv")
    trainable_params = sum(p.numel() for p in model.get_trainable_parameters() if p.requires_grad)

    log_msg("-" * 80, log_file)
    log_msg("Commencing Test Split Evaluation Across SNR Grid", log_file)
    log_msg(f"Evaluation SNR Grid: {snr_grid} dB", log_file)
    log_msg("-" * 80, log_file)

    snr_rows = []

    for snr_db in snr_grid:
        test_loader.dataset.set_snr(float(snr_db))
        all_targets = {t: [] for t in range(5)}
        all_preds = {t: [] for t in range(5)}

        with torch.no_grad():
            for batch in test_loader:
                snr_t = batch.get('snr')
                if snr_t is not None:
                    snr_t = snr_t.to(device)

                for t in range(5):
                    inputs = batch['inputs'][:, t].to(device)
                    target = batch[f'target_{t}']
                    B_size = inputs.shape[0]
                    task_ids = torch.full((B_size,), t, dtype=torch.long, device=device)

                    if is_a3:
                        out = model(inputs, task_ids, snr=snr_t)
                    else:
                        scene = batch.get('scene')
                        if scene is not None:
                            scene = scene.to(device)
                        out = model(inputs, scene, task_ids, hard=True, snr=snr_t)

                    all_preds[t].append(out[f'out_{t}'].cpu().numpy())
                    all_targets[t].append(target.numpy())

        preds = {t: np.concatenate(all_preds[t], axis=0) for t in range(5)}
        targets = {t: np.concatenate(all_targets[t], axis=0) for t in range(5)}

        ce_nmse = evaluate_channel_estimation_nmse(preds[0], targets[0])
        det_ber, det_ser = evaluate_mimo_detection_ber(preds[1], targets[1])
        prec_rate = evaluate_precoding_sum_rate(preds[2], targets[0], snr_db=snr_db)
        dec_ber, dec_bler = evaluate_channel_decoding(preds[3], targets[3])
        loc_rmse = evaluate_user_localization_rmse(preds[4], targets[4])

        log_msg(
            f"SNR: {snr_db:3d} dB | CE NMSE: {ce_nmse:7.2f} dB | "
            f"DET_BER_MMSE_init: {det_ber:.4f} | PREC Rate: {prec_rate:.2f} bps/Hz | "
            f"DEC BER: {dec_ber:.4f} | LOC RMSE: {loc_rmse:.2f} m",
            log_file
        )

        snr_rows.append({
            "SNR_dB": int(snr_db),
            "CE_NMSE_dB": round(ce_nmse, 4),
            "DET_BER_MMSE_init": round(det_ber, 8),
            "PREC_SumRate": round(prec_rate, 4),
            "DEC_BER": round(dec_ber, 8),
            "LOC_RMSE_m": round(loc_rmse, 4),
            "Trainable_Params": trainable_params
        })

    with open(snr_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "SNR_dB",
            "CE_NMSE_dB",
            "DET_BER_MMSE_init",
            "PREC_SumRate",
            "DEC_BER",
            "LOC_RMSE_m",
            "Trainable_Params"
        ])
        for r in snr_rows:
            writer.writerow([
                r["SNR_dB"],
                r["CE_NMSE_dB"],
                r["DET_BER_MMSE_init"],
                r["PREC_SumRate"],
                r["DEC_BER"],
                r["LOC_RMSE_m"],
                r["Trainable_Params"]
            ])
    log_msg(f"Saved SNR evaluation results to: {snr_csv_path}", log_file)


def train_ablation(args):
    ablation_key = args.ablation
    env_name = args.env
    epochs = args.epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = os.path.join("logs", "day13", ablation_key)
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "run_log.txt")
    metrics_csv = os.path.join(output_dir, "metrics.csv")
    
    ckpt_dir = os.path.join("results", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, f"day13_{ablation_key}_best.pt")

    cfg = load_config(args.config)
    set_seed(cfg['system']['seed'])

    log_msg("=" * 80, log_file)
    log_msg(f"DAY 13 ABLATION STUDY: {ablation_key}", log_file)
    log_msg(f"Environment: {env_name} | Device: {device} | Epochs: {epochs}", log_file)
    log_msg("=" * 80, log_file)

    # 1. Instantiate Target Ablation Architecture
    is_a3 = False
    if ablation_key == "A1_no_scene":
        model = AblationA1_NoScene(
            ranks=cfg['model']['rank_set'],
            feature_dim=cfg['wireless']['feature_dim'],
            semantic_coupling=False,
            lora_alpha=1.0
        ).to(device)
        desc = "Ablation A1: No Scene Encoder (ViT removed, zero-filled scene vector)"
    elif ablation_key == "A2_no_adaptive_rank":
        model = AblationA2_NoAdaptiveRank(
            ranks=cfg['model']['rank_set'],
            feature_dim=cfg['wireless']['feature_dim'],
            semantic_coupling=False,
            lora_alpha=1.0
        ).to(device)
        desc = "Ablation A2: No Adaptive Rank (Forced rank=8, no Gumbel rank gate)"
    elif ablation_key == "A3_no_hypernet":
        model = AblationA3_NoHypernet(
            rank=8,
            feature_dim=cfg['wireless']['feature_dim'],
            semantic_coupling=False
        ).to(device)
        is_a3 = True
        desc = "Ablation A3: No Hypernetwork (Static LoRA r=8 with same task heads and recipe)"
    else:
        raise ValueError(f"Unknown ablation key: {ablation_key}")

    log_msg(f"Description: {desc}", log_file)

    trainable_params = model.get_trainable_parameters()
    num_trainable = sum(p.numel() for p in trainable_params if p.requires_grad)
    log_msg(f"Trainable Parameters: {num_trainable:,}", log_file)

    # Save configuration snapshot
    save_config(cfg, os.path.join(output_dir, "config.yaml"))

    # 2. Data Loaders
    train_loader, val_loader, test_loader = get_dataloaders(
        env_name=env_name,
        data_dir=cfg['data']['data_dir'],
        batch_size=cfg['training']['batch_size'],
        train_split=cfg['data']['train_split'],
        val_split=cfg['data']['val_split'],
        max_samples=args.max_samples,
        seed=cfg['system']['seed']
    )

    optimizer = optim.AdamW(trainable_params, lr=1e-3 if is_a3 else 5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Initialize metrics CSV
    with open(metrics_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "CE_NMSE_dB",
            "DET_BER_MMSE_init",
            "PREC_SumRate",
            "DEC_BER",
            "LOC_RMSE_m",
            "trainable_params",
            "runtime_s"
        ])

    best_val_loss = float("inf")

    # 3. Training Loop (10 epochs)
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            batch_size = batch['inputs'].shape[0]
            step_loss = 0.0

            scene = batch.get('scene')
            if scene is not None and not is_a3:
                scene = scene.to(device)

            snr_t = batch.get('snr')
            if snr_t is not None:
                snr_t = snr_t.to(device)

            for t in range(5):
                inputs = batch['inputs'][:, t].to(device)
                target = batch[f'target_{t}'].to(device)
                task_ids = torch.full((batch_size,), t, dtype=torch.long, device=device)

                if is_a3:
                    out = model(inputs, task_ids, snr=snr_t)
                else:
                    out = model(inputs, scene, task_ids, hard=True, snr=snr_t)

                pred = out[f'out_{t}']
                true_ch = batch['target_0'].to(device) if 'target_0' in batch else None

                t_loss = compute_task_loss(
                    t,
                    pred,
                    target,
                    inputs=inputs,
                    true_channels=true_ch,
                    snr=snr_t
                )
                step_loss = step_loss + (t_loss / 5.0)

            step_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            total_train_loss += float(step_loss.item())
            num_batches += 1

        scheduler.step()
        epoch_train_loss = total_train_loss / max(num_batches, 1)

        # Validation evaluation at reference 10 dB
        model.eval()
        val_loader.dataset.set_snr(10.0)
        v_preds = {t: [] for t in range(5)}
        v_targets = {t: [] for t in range(5)}

        with torch.no_grad():
            for batch in val_loader:
                snr_t = batch.get('snr')
                if snr_t is not None:
                    snr_t = snr_t.to(device)

                for t in range(5):
                    inputs = batch['inputs'][:, t].to(device)
                    target = batch[f'target_{t}']
                    task_ids = torch.full((inputs.shape[0],), t, dtype=torch.long, device=device)

                    if is_a3:
                        out = model(inputs, task_ids, snr=snr_t)
                    else:
                        scene = batch.get('scene')
                        if scene is not None:
                            scene = scene.to(device)
                        out = model(inputs, scene, task_ids, hard=True, snr=snr_t)

                    v_preds[t].append(out[f'out_{t}'].cpu().numpy())
                    v_targets[t].append(target.numpy())

        vp = {t: np.concatenate(v_preds[t], axis=0) for t in range(5)}
        vt = {t: np.concatenate(v_targets[t], axis=0) for t in range(5)}

        val_ce = evaluate_channel_estimation_nmse(vp[0], vt[0])
        val_det, _ = evaluate_mimo_detection_ber(vp[1], vt[1])
        val_prec = evaluate_precoding_sum_rate(vp[2], vt[0], snr_db=10.0)
        val_dec, _ = evaluate_channel_decoding(vp[3], vt[3])
        val_loc = evaluate_user_localization_rmse(vp[4], vt[4])
        epoch_runtime = time.time() - epoch_start

        log_msg(
            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.4f} | "
            f"Val -> CE: {val_ce:6.2f} dB | DET_BER_MMSE_init: {val_det:.4f} | "
            f"PREC: {val_prec:5.2f} bps/Hz | DEC: {val_dec:.4f} | LOC: {val_loc:5.2f} m | "
            f"Runtime: {epoch_runtime:.1f}s",
            log_file
        )

        with open(metrics_csv, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                round(epoch_train_loss, 4),
                round(val_ce, 4),
                round(val_det, 6),
                round(val_prec, 4),
                round(val_dec, 6),
                round(val_loc, 4),
                num_trainable,
                round(epoch_runtime, 2)
            ])

        if epoch_train_loss < best_val_loss:
            best_val_loss = epoch_train_loss
            torch.save(model.state_dict(), best_ckpt_path)

    log_msg(f"Training complete. Best model checkpoint saved to: {best_ckpt_path}", log_file)

    # 4. Checkpoint Evaluation on Unseen Test Split
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device, weights_only=False))
    model.eval()
    evaluate_ablation_checkpoint_snr(
        model=model,
        test_loader=test_loader,
        output_dir=output_dir,
        device=device,
        is_a3=is_a3,
        log_file=log_file
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 13 Ablation Training Runner")
    parser.add_argument("--ablation", type=str, required=True, choices=["A1_no_scene", "A2_no_adaptive_rank", "A3_no_hypernet"])
    parser.add_argument("--env", type=str, default="Urban_Macro")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    train_ablation(args)
