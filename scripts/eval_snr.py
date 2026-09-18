"""
scripts/eval_snr.py
Evaluates a trained checkpoint across the official SNR grid (-5 to 25 dB).
Outputs the official snr_evaluation.csv file with canonical columns.

Usage:
  python scripts/eval_snr.py --ckpt <path> --out logs/<dir>/snr_evaluation.csv
"""
import os
import sys
import csv
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.seed import set_seed
from src.utils.config import load_config
from src.data.dataset import get_dataloaders
from src.models.wireless_fm import EnvironmentAwareWirelessFM
from src.models.baselines import StaticLoRAModel
from src.models.ablations import AblationA1_NoScene, AblationA2_NoAdaptiveRank, AblationA3_NoHypernet
from src.eval.metrics import (
    evaluate_channel_estimation_nmse,
    evaluate_mimo_detection_ber,
    evaluate_precoding_sum_rate,
    evaluate_channel_decoding,
    evaluate_user_localization_rmse
)


def load_model_from_checkpoint(ckpt_path: str, cfg: dict, device: str = "cpu"):
    """Auto-detects model type from state_dict keys and instantiates appropriate model."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    keys = list(state_dict.keys())

    # Detect architecture based on parameter signatures
    is_a1 = any("hypernetwork" in k for k in keys) and not any("scene_encoder" in k for k in keys)
    is_a2 = any("fixed_lora_A" in k for k in keys)
    is_a3 = any("lora_A" in k for k in keys) and not any("hypernetwork" in k for k in keys)
    is_static = any("lora_A" in k for k in keys) and not any("hypernetwork" in k for k in keys)
    is_proposed = any("hypernetwork" in k for k in keys) and any("scene_encoder" in k for k in keys)

    if "A1" in ckpt_path or is_a1:
        print("-> Detected Architecture: Ablation A1 (No Scene Conditioning)")
        model = AblationA1_NoScene(ranks=cfg['model']['rank_set'], feature_dim=cfg['wireless']['feature_dim']).to(device)
        model_type = "ablation_a1"
    elif "A2" in ckpt_path or is_a2:
        print("-> Detected Architecture: Ablation A2 (No Adaptive Rank, Fixed r=8)")
        model = AblationA2_NoAdaptiveRank(rank=8, feature_dim=cfg['wireless']['feature_dim']).to(device)
        model_type = "ablation_a2"
    elif "A3" in ckpt_path:
        print("-> Detected Architecture: Ablation A3 (No Hypernetwork)")
        model = AblationA3_NoHypernet(rank=8, feature_dim=cfg['wireless']['feature_dim']).to(device)
        model_type = "ablation_a3"
    elif "static" in ckpt_path.lower() or is_static:
        print("-> Detected Architecture: Static LoRA (Fixed r=8, No Hypernetwork)")
        model = StaticLoRAModel(rank=8).to(device)
        model_type = "static_lora"
    else:
        print("-> Detected Architecture: Proposed Dynamic LoRA (Hypernetwork + Rank Gate)")
        model = EnvironmentAwareWirelessFM(
            ranks=cfg['model']['rank_set'],
            feature_dim=cfg['wireless']['feature_dim'],
            semantic_coupling=cfg['model'].get('semantic_coupling', False),
            lora_alpha=cfg['model'].get('lora_alpha', 1.0)
        ).to(device)
        model_type = "proposed"

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, model_type


def evaluate_snr_grid(
    ckpt_path: str,
    out_path: str,
    env_name: str = "Urban_Macro",
    config_path: str = "configs/default.yaml",
    snr_grid=None
):
    if snr_grid is None:
        snr_grid = [-5, 0, 5, 10, 15, 20, 25]

    cfg = load_config(config_path)
    set_seed(cfg["system"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print(f"OFFICIAL SNR EVALUATION: {os.path.basename(ckpt_path)}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Target Output: {out_path}")
    print(f"Environment: {env_name} | Device: {device} | SNR Grid: {snr_grid}")
    print("=" * 80)

    model, model_type = load_model_from_checkpoint(ckpt_path, cfg, device)

    if hasattr(model, 'get_trainable_parameters'):
        trainable_params = sum(p.numel() for p in model.get_trainable_parameters() if p.requires_grad)
    else:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Trainable Parameters: {trainable_params:,}")

    # Load test split
    _, _, test_loader = get_dataloaders(
        env_name=env_name,
        data_dir=cfg["data"]["data_dir"],
        batch_size=cfg["training"]["batch_size"],
        train_split=cfg["data"]["train_split"],
        val_split=cfg["data"]["val_split"],
        seed=cfg["system"]["seed"]
    )

    rows = []

    for snr_db in snr_grid:
        test_loader.dataset.set_snr(float(snr_db))
        all_preds = {t: [] for t in range(5)}
        all_targets = {t: [] for t in range(5)}

        with torch.no_grad():
            for batch in test_loader:
                snr_tensor = batch.get('snr')
                if snr_tensor is not None:
                    snr_tensor = snr_tensor.to(device)

                for t in range(5):
                    inputs = batch['inputs'][:, t].to(device)
                    target = batch[f'target_{t}']
                    B_size = inputs.shape[0]
                    task_ids = torch.full((B_size,), t, dtype=torch.long, device=device)

                    if model_type in ["static_lora", "ablation_a3"]:
                        out = model(inputs, task_ids, snr=snr_tensor)
                    elif model_type == "ablation_a1":
                        scene = batch['scene'].to(device)
                        out = model(inputs, scene, task_ids, hard=True, snr=snr_tensor)
                    else:
                        scene = batch['scene'].to(device)
                        cls_emb = model.scene_encoder(scene)
                        z_s = model.scene_proj(cls_emb)
                        out = model(inputs, scene, task_ids, hard=True, z_s=z_s, snr=snr_tensor)

                    all_preds[t].append(out[f'out_{t}'].cpu().numpy())
                    all_targets[t].append(target.numpy())

        preds = {t: np.concatenate(all_preds[t], axis=0) for t in range(5)}
        targets = {t: np.concatenate(all_targets[t], axis=0) for t in range(5)}

        ce_nmse = evaluate_channel_estimation_nmse(preds[0], targets[0])
        det_ber, det_ser = evaluate_mimo_detection_ber(preds[1], targets[1])
        prec_rate = evaluate_precoding_sum_rate(preds[2], targets[0], snr_db=snr_db)
        dec_ber, _ = evaluate_channel_decoding(preds[3], targets[3])
        loc_rmse = evaluate_user_localization_rmse(preds[4], targets[4])

        print(
            f"SNR: {snr_db:>2d} dB | CE NMSE: {ce_nmse:>6.2f} dB | "
            f"DET BER: {det_ber:.4f} (SER: {det_ser:.4f}) | "
            f"PREC Sum-Rate: {prec_rate:>5.2f} bps/Hz | "
            f"DEC BER: {dec_ber:.4f} | LOC RMSE: {loc_rmse:>5.2f} m"
        )

        rows.append({
            "SNR_dB": int(snr_db),
            "CE_NMSE_dB": round(ce_nmse, 4),
            "DET_BER": round(det_ber, 4),
            "DET_SER": round(det_ser, 4),
            "PREC_SumRate": round(prec_rate, 4),
            "DEC_BER": round(dec_ber, 4),
            "LOC_RMSE_m": round(loc_rmse, 4),
            "Trainable_Params": trainable_params
        })

    # Save to out_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = ["SNR_dB", "CE_NMSE_dB", "DET_BER", "DET_SER", "PREC_SumRate", "DEC_BER", "LOC_RMSE_m", "Trainable_Params"]
    with open(out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\n[✓] Saved official evaluation metrics to: {out_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Official SNR Grid Evaluation")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--out", type=str, required=True, help="Path to output snr_evaluation.csv")
    parser.add_argument("--env", type=str, default="Urban_Macro", help="Environment (default: Urban_Macro)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    evaluate_snr_grid(
        ckpt_path=args.ckpt,
        out_path=args.out,
        env_name=args.env,
        config_path=args.config
    )


if __name__ == "__main__":
    main()
