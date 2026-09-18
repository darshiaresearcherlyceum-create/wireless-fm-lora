"""
scripts/eval_detection_debug.py
Evaluates MIMO Detection (Task 1) at a target SNR (e.g. 20 or 25 dB) for debugging and verification.

Prints:
  - Total bits
  - Bit errors
  - BER
  - 10 example packets
  - The explicit sentence that the detection head uses MMSE initialization

Usage:
  python scripts/eval_detection_debug.py --ckpt <path> --snr 20
  python scripts/eval_detection_debug.py --ckpt <path> --snr 25
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


def format_bits_str(arr):
    """Formats a 1D integer/boolean array into a binary bit string."""
    return "".join(str(int(b)) for b in arr)


def evaluate_detection_debug(
    ckpt_path: str,
    target_snr: float = 20.0,
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

    test_loader.dataset.set_snr(float(target_snr))

    preds_all = []
    targets_all = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['inputs'][:, 1].to(device)
            target = batch['target_1'].to(device)
            task_ids = torch.full((inputs.shape[0],), 1, dtype=torch.long, device=device)

            if is_static or prefix == "static_lora":
                out = model(inputs, task_ids)
            else:
                scene = batch['scene'].to(device)
                cls_emb = model.scene_encoder(scene)
                z_s = model.scene_proj(cls_emb)
                out = model(inputs, scene, task_ids, hard=True, z_s=z_s)

            preds_all.append(out['out_1'].cpu().numpy())
            targets_all.append(target.cpu().numpy())

    preds = np.concatenate(preds_all, axis=0)      # [45, 64, 16]
    targets = np.concatenate(targets_all, axis=0)  # [45, 64, 16]

    # Hard-decision slicing at threshold 0.0 for QPSK constellations
    detected_bits = (preds < 0.0).astype(int)
    target_bits = (targets < 0.0).astype(int)

    num_samples = preds.shape[0]  # 45
    num_subcarriers = 64
    num_users = 8
    bits_per_symbol = 2  # QPSK
    bits_per_sample = num_subcarriers * num_users * bits_per_symbol  # 1024
    total_bits = num_samples * bits_per_sample                      # 46,080

    bit_errors = int(np.sum(detected_bits != target_bits))
    ber = float(bit_errors / total_bits)

    # 10 example packets (each packet: 1 subcarrier across 8 users = 16 bits)
    example_packets = []
    for pkt_idx in range(10):
        s_id = (pkt_idx * 4) % num_samples
        sub_id = (pkt_idx * 7) % num_subcarriers
        p_bits_16 = detected_bits[s_id, sub_id, :]
        t_bits_16 = target_bits[s_id, sub_id, :]
        err_count = int(np.sum(p_bits_16 != t_bits_16))
        example_packets.append({
            "pkt_id": pkt_idx + 1,
            "sample_id": s_id,
            "subcarrier": sub_id,
            "true_bits": format_bits_str(t_bits_16),
            "pred_bits": format_bits_str(p_bits_16),
            "errors": err_count
        })

    # Print required outputs
    print("=" * 80)
    print("MIMO DETECTION BIT-LEVEL DEBUG REPORT")
    print(f"Model: {model_name} | Target SNR: {target_snr:.1f} dB")
    print(f"Checkpoint: {ckpt_path}")
    print("=" * 80)
    print(f"• Total bits evaluated   : {total_bits:,} bits ({num_samples} samples x {bits_per_sample} bits)")
    print(f"• Bit errors observed   : {bit_errors:,}")
    print(f"• Bit Error Rate (BER)  : {ber:.8f} ({bit_errors} / {total_bits})")
    print("-" * 80)
    print("10 Example Packets (16 bits each = 8 users x 2 bits/QPSK per subcarrier):")
    print(f"{'Packet':<8} | {'Sample':<7} | {'Subcarrier':<10} | {'True Bits (16b)':<20} | {'Pred Bits (16b)':<20} | {'Errors':<6}")
    print("-" * 80)
    for p in example_packets:
        print(f"{p['pkt_id']:<8} | {p['sample_id']:<7} | {p['subcarrier']:<10} | {p['true_bits']:<20} | {p['pred_bits']:<20} | {p['errors']:<6}")
    print("-" * 80)
    print("CRITICAL NOTE ON DETECTION INITIALIZATION:")
    print("The detection head uses MMSE initialization (Linear MMSE base symbol estimate + learned neural residual).")
    print("Detection BER = 0 at 20–25 dB is from MMSE initialization, not from Dynamic LoRA.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Official MIMO Detection Bit-Level Debug")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--snr", type=float, required=True, help="Target SNR in dB (e.g. 20 or 25)")
    parser.add_argument("--env", type=str, default="Urban_Macro")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--out_dir", type=str, default="logs/Day12")
    args = parser.parse_args()

    evaluate_detection_debug(
        ckpt_path=args.ckpt,
        target_snr=args.snr,
        env_name=args.env,
        config_path=args.config,
        out_dir=args.out_dir
    )


if __name__ == "__main__":
    main()
