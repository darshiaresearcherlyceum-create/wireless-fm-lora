"""
scripts/run_rzf_baseline.py
Evaluates the classical Regularized Zero-Forcing (RZF) baseline precoder.
Pure mathematical / analytical baseline:
- No neural network / No GPT-2 / No LoRA / 0 trainable parameters.
- Computes and saves only Precoding sum-rate (bps/Hz) across the SNR grid.
"""
import os
import sys
import csv
import datetime
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.seed import set_seed
from src.utils.config import load_config
from src.data.dataset import get_dataloaders
from src.eval.metrics import compute_classical_rzf_precoder, evaluate_precoding_sum_rate


def log_msg(msg: str, log_file: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"{timestamp} [INFO] {msg}"
    print(formatted, flush=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")


def run_rzf_evaluation(
    env_name: str = "Urban_Macro",
    output_dir: str = "logs/day11/rzf_baseline",
    snr_grid=None,
    alpha: float = 0.018,
    config_path: str = "configs/default.yaml"
):
    if snr_grid is None:
        snr_grid = [-5, 0, 5, 10, 15, 20, 25]

    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "run_log.txt")
    snr_csv_file = os.path.join(output_dir, "snr_evaluation.csv")

    cfg = load_config(config_path)
    set_seed(cfg["system"]["seed"])

    log_msg("=" * 80, log_file)
    log_msg("METHOD: RZF_BASELINE (Classical Precoding Only)", log_file)
    log_msg("Architecture: Classical Closed-Form RZF (No GPT-2, No LoRA, 0 Trainable Parameters)", log_file)
    log_msg(f"Trainable Parameters: 0", log_file)
    log_msg(f"Environment: {env_name}", log_file)
    log_msg(f"Evaluation SNR Grid: {snr_grid} dB", log_file)
    log_msg("=" * 80, log_file)

    _, _, test_loader = get_dataloaders(
        env_name=env_name,
        data_dir=cfg["data"]["data_dir"],
        batch_size=cfg["training"]["batch_size"],
        train_split=cfg["data"]["train_split"],
        val_split=cfg["data"]["val_split"],
        seed=cfg["system"]["seed"]
    )

    results = []

    for snr_db in snr_grid:
        test_loader.dataset.set_snr(float(snr_db))

        all_channels = []
        all_rzf_w = []

        for batch in test_loader:
            channels = batch["target_0"].numpy()  # Ground-truth channels for downlink beamforming
            w_rzf = compute_classical_rzf_precoder(channels, alpha=alpha)
            all_channels.append(channels)
            all_rzf_w.append(w_rzf)

        channels_arr = np.concatenate(all_channels, axis=0)
        rzf_w_arr = np.concatenate(all_rzf_w, axis=0)

        prec_sum_rate = evaluate_precoding_sum_rate(rzf_w_arr, channels_arr, snr_db=snr_db)

        row = {
            "SNR_dB": int(snr_db),
            "CE_NMSE_dB": "N/A",
            "DET_BER": "N/A",
            "DET_SER": "N/A",
            "PREC_SumRate": round(prec_sum_rate, 4),
            "DEC_BER": "N/A",
            "LOC_RMSE_m": "N/A",
            "Trainable_Params": 0
        }
        results.append(row)
        log_msg(f"SNR: {snr_db:>2d} dB | PREC Sum-Rate: {row['PREC_SumRate']:>6.2f} bps/Hz", log_file)

    # Save snr_evaluation.csv
    fieldnames = ["SNR_dB", "CE_NMSE_dB", "DET_BER", "DET_SER", "PREC_SumRate", "DEC_BER", "LOC_RMSE_m", "Trainable_Params"]
    with open(snr_csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    log_msg(f"Saved SNR evaluation metrics to: {snr_csv_file}", log_file)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Classical RZF Precoding Baseline")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--env", type=str, default="Urban_Macro")
    parser.add_argument("--output_dir", type=str, default="logs/day11/rzf_baseline")
    parser.add_argument("--alpha", type=float, default=0.018, help="RZF regularization parameter")
    args = parser.parse_args()

    run_rzf_evaluation(
        env_name=args.env,
        output_dir=args.output_dir,
        alpha=args.alpha,
        config_path=args.config
    )


if __name__ == "__main__":
    main()
