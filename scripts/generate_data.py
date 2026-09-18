"""
scripts/generate_data.py
Dataset generation script for the official Urban_Macro wireless experiment.
Generates synthetic MIMO-OFDM channels following 3GPP TR 38.901 propagation models.
(Note: Uses pure NumPy implementation of 3GPP TR 38.901 spatial channel model;
 no external DeepMIMO or Sionna dependencies are used).
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.generation import generate_channel_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate official Urban_Macro dataset")
    parser.add_argument("--env", type=str, default="Urban_Macro", choices=["Urban_Macro"], help="Official environment (Urban_Macro only)")
    parser.add_argument("-n", "--n", "--samples", dest="samples", type=int, default=300, help="Total number of samples (default: 300)")
    parser.add_argument("--output_dir", type=str, default="dataset_wireless", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--antennas", type=int, default=64, help="Number of BS antennas (default: 64)")
    parser.add_argument("--users", type=int, default=8, help="Number of users (default: 8)")
    parser.add_argument("--subcarriers", type=int, default=64, help="Number of subcarriers (default: 64)")
    args = parser.parse_args()

    num_samples = args.samples
    num_antennas = args.antennas
    num_users = args.users
    num_subcarriers = args.subcarriers
    seed = args.seed
    output_dir = args.output_dir
    env_name = args.env

    # 210 / 45 / 45 split calculation
    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15

    num_train = int(train_ratio * num_samples)
    num_val = int(val_ratio * num_samples)
    num_test = num_samples - num_train - num_val

    snr_grid = [-5, 0, 5, 10, 15, 20, 25]

    print("=" * 65)
    print(f"Official Experiment Data Generation: {env_name}")
    print("=" * 65)
    print(f"  • Environment:     {env_name} (3GPP TR 38.901 UMa)")
    print(f"  • Total Samples:   {num_samples}")
    print(f"  • Split:           {num_train} train / {num_val} val / {num_test} test")
    print(f"  • Geometry:        {num_antennas} BS antennas, {num_users} users, {num_subcarriers} subcarriers")
    print(f"  • Modulation:      QPSK")
    print(f"  • SNR Grid (dB):   {snr_grid}")
    print(f"  • Seed:            {seed}")
    print("=" * 65)

    save_path = generate_channel_dataset(
        env_name=env_name,
        output_dir=output_dir,
        num_samples=num_samples,
        num_users=num_users,
        num_subcarriers=num_subcarriers,
        num_bs_antennas=num_antennas,
        seed=seed
    )

    # Verification of saved file
    data = np.load(save_path)
    total_saved = len(data["channel"])
    print("\nGeneration Summary:")
    print(f"  [✓] Successfully saved {total_saved} samples to: {save_path}")
    print(f"  [✓] Data keys: {list(data.keys())}")
    print(f"  [✓] Channel tensor shape: {data['channel'].shape}")
    print(f"  [✓] Dataset split: {num_train} train / {num_val} val / {num_test} test")
    print("=" * 65)


if __name__ == "__main__":
    main()
