"""
scripts/eval_localization_range.py
Evaluates user 3D localization range and test RMSE from a trained checkpoint.
Generates statistical range summary and example predictions vs true coordinates.

Usage:
  python scripts/eval_localization_range.py --ckpt <path>
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


def evaluate_localization_range(
    ckpt_path: str,
    env_name: str = "Urban_Macro",
    config_path: str = "configs/default.yaml",
    out_dir: str = "logs/Day12"
):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    os.makedirs(out_dir, exist_ok=True)
    cfg = load_config(config_path)
    set_seed(cfg["system"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    is_static = any("lora_A" in k for k in state_dict.keys()) and not any("hypernetwork" in k for k in state_dict.keys())

    if is_static or "static" in ckpt_path.lower():
        model_name = "Static LoRA (r=8 Baseline)"
        prefix = "static_lora"
        model = StaticLoRAModel(rank=8).to(device)
    else:
        model_name = "Proposed Dynamic LoRA"
        prefix = "proposed"
        model = EnvironmentAwareWirelessFM(
            ranks=cfg['model']['rank_set'],
            feature_dim=cfg['wireless']['feature_dim'],
            semantic_coupling=cfg['model'].get('semantic_coupling', False),
            lora_alpha=1.0
        ).to(device)

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Load test split
    _, _, test_loader = get_dataloaders(
        env_name=env_name,
        data_dir=cfg["data"]["data_dir"],
        batch_size=cfg["training"]["batch_size"],
        train_split=cfg["data"]["train_split"],
        val_split=cfg["data"]["val_split"],
        seed=cfg["system"]["seed"]
    )

    preds_list = []
    targets_list = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['inputs'][:, 4].to(device)
            target = batch['target_4']
            task_ids = torch.full((inputs.shape[0],), 4, dtype=torch.long, device=device)

            if is_static or prefix == "static_lora":
                out = model(inputs, task_ids)
            else:
                scene = batch['scene'].to(device)
                cls_emb = model.scene_encoder(scene)
                z_s = model.scene_proj(cls_emb)
                out = model(inputs, scene, task_ids, hard=True, z_s=z_s)

            preds_list.append(out['out_4'].cpu().numpy())
            targets_list.append(target.numpy())

    # Shape: [N_samples, 8, 3] in meters (0-1 scaled by 100m arena)
    p_flat = np.concatenate(preds_list, axis=0).reshape(-1, 8, 3).reshape(-1, 3) * 100.0
    t_flat = np.concatenate(targets_list, axis=0).reshape(-1, 8, 3).reshape(-1, 3) * 100.0

    # Coordinate stats
    coords = ["X", "Y", "Z"]
    stats = {}
    for i, c in enumerate(coords):
        pred_c = p_flat[:, i]
        true_c = t_flat[:, i]
        rmse_c = float(np.sqrt(np.mean((pred_c - true_c) ** 2)))
        stats[c] = {
            "pred_min": float(np.min(pred_c)),
            "pred_max": float(np.max(pred_c)),
            "pred_mean": float(np.mean(pred_c)),
            "pred_std": float(np.std(pred_c)),
            "true_min": float(np.min(true_c)),
            "true_max": float(np.max(true_c)),
            "true_mean": float(np.mean(true_c)),
            "true_std": float(np.std(true_c)),
            "rmse": rmse_c
        }

    overall_3d_rmse = float(np.sqrt(np.mean(np.sum((p_flat - t_flat) ** 2, axis=-1))))

    # 1. Print report to stdout
    print("=" * 95)
    print("PART C: LOCALIZATION PREDICTED VS TRUE RANGE REPORT")
    print(f"Model: {model_name} | Checkpoint: {ckpt_path}")
    print(f"Total Test Users Evaluated: {len(p_flat)} (45 test samples x 8 users)")
    print(f"Overall 3D Euclidean Test RMSE: {overall_3d_rmse:.4f} meters")
    print("=" * 95)
    print("--- STATISTICAL RANGE SUMMARY (METERS) ---")
    print(f"{'Coord':<5} | {'Pred Min':<9} | {'Pred Max':<9} | {'Pred Mean':<9} | {'Pred Std':<9} | {'True Min':<9} | {'True Max':<9} | {'True Mean':<9} | {'True Std':<9} | {'RMSE (m)':<9}")
    print("-" * 105)
    for c in coords:
        s = stats[c]
        print(f"{c:<5} | {s['pred_min']:<9.4f} | {s['pred_max']:<9.4f} | {s['pred_mean']:<9.4f} | {s['pred_std']:<9.4f} | {s['true_min']:<9.4f} | {s['true_max']:<9.4f} | {s['true_mean']:<9.4f} | {s['true_std']:<9.4f} | {s['rmse']:<9.4f}")
    print("-" * 105)
    print(f"Overall 3D Euclidean RMSE: {overall_3d_rmse:.4f} m\n")

    print("=" * 95)
    print("10 EXAMPLE USERS: TRUE VS PREDICTED (X, Y, Z) AND 3D ERROR DISTANCE")
    print("=" * 95)
    print(f"{'User #':<7} | {'True (x, y, z) in meters':<32} | {'Predicted (x, y, z) in meters':<34} | {'Error (m)':<10}")
    print("-" * 95)
    for u in range(10):
        t_xyz = f"({t_flat[u, 0]:6.2f}, {t_flat[u, 1]:6.2f}, {t_flat[u, 2]:6.2f})"
        p_xyz = f"({p_flat[u, 0]:6.2f}, {p_flat[u, 1]:6.2f}, {p_flat[u, 2]:6.2f})"
        err = float(np.sqrt(np.sum((p_flat[u] - t_flat[u]) ** 2)))
        print(f"{u+1:<7} | {t_xyz:<32} | {p_xyz:<34} | {err:<10.4f}")
    print("=" * 95)

    # 2. Save CSV
    csv_file = os.path.join(out_dir, f"{prefix}_localization_range.csv")
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Coordinate", "Pred_Min_m", "Pred_Max_m", "Pred_Mean_m", "Pred_Std_m", "True_Min_m", "True_Max_m", "True_Mean_m", "True_Std_m", "Test_RMSE_m"])
        for c in coords:
            s = stats[c]
            writer.writerow([c, f"{s['pred_min']:.4f}", f"{s['pred_max']:.4f}", f"{s['pred_mean']:.4f}", f"{s['pred_std']:.4f}", f"{s['true_min']:.4f}", f"{s['true_max']:.4f}", f"{s['true_mean']:.4f}", f"{s['true_std']:.4f}", f"{s['rmse']:.4f}"])
        writer.writerow(["Overall_3D", "-", "-", "-", "-", "-", "-", "-", "-", f"{overall_3d_rmse:.4f}"])

    # 3. Save Text Summary
    txt_file = os.path.join(out_dir, f"{prefix}_localization_examples.txt")
    with open(txt_file, mode="w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("PART C: LOCALIZATION PREDICTED VS TRUE RANGE REPORT\n")
        f.write(f"METHOD: {model_name}\n")
        f.write(f"OVERALL 3D TEST RMSE: {overall_3d_rmse:.4f} meters\n")
        f.write("=" * 80 + "\n\n")
        f.write("--- STATISTICAL RANGE SUMMARY (METERS) ---\n")
        f.write("Coord | Pred Min  | Pred Max  | Pred Mean  | Pred Std  | True Min  | True Max  | True Mean  | True Std  | RMSE (m)\n")
        f.write("-" * 105 + "\n")
        for c in coords:
            s = stats[c]
            f.write(f"{c:<5} | {s['pred_min']:<9.4f} | {s['pred_max']:<9.4f} | {s['pred_mean']:<9.4f} | {s['pred_std']:<9.4f} | {s['true_min']:<9.4f} | {s['true_max']:<9.4f} | {s['true_mean']:<9.4f} | {s['true_std']:<9.4f} | {s['rmse']:<9.4f}\n")
        f.write("-" * 105 + "\n")
        f.write(f"Overall 3D Euclidean RMSE: {overall_3d_rmse:.4f} m\n\n")
        f.write("=" * 80 + "\n")
        f.write("20 EXAMPLE USERS: TRUE VS PREDICTED (X, Y, Z) AND 3D ERROR DISTANCE\n")
        f.write("=" * 80 + "\n")
        f.write("User #  | True (x, y, z) in meters         | Predicted (x, y, z) in meters      | Error (m) \n")
        f.write("-" * 90 + "\n")
        for u in range(min(20, len(p_flat))):
            t_xyz = f"({t_flat[u, 0]:6.2f}, {t_flat[u, 1]:6.2f}, {t_flat[u, 2]:6.2f})"
            p_xyz = f"({p_flat[u, 0]:6.2f}, {p_flat[u, 1]:6.2f}, {p_flat[u, 2]:6.2f})"
            err = float(np.sqrt(np.sum((p_flat[u] - t_flat[u]) ** 2)))
            f.write(f"{u+1:<7} | {t_xyz:<32} | {p_xyz:<34} | {err:<10.4f}\n")

    print(f"[✓] Saved localization range report to: {csv_file}")
    print(f"[✓] Saved detailed examples to: {txt_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Localization Range and Test RMSE")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--env", type=str, default="Urban_Macro")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--out_dir", type=str, default="logs/Day12")
    args = parser.parse_args()

    evaluate_localization_range(
        ckpt_path=args.ckpt,
        env_name=args.env,
        config_path=args.config,
        out_dir=args.out_dir
    )


if __name__ == "__main__":
    main()
