"""Run all Phase 1 baseline experiments sequentially.

Phase 1 = ResNet-50 / P0 / 100% data / Full Fine-Tune on both datasets,
3 seeds each. This reproduces the MedMNIST published benchmark and validates
the harness before scaling to the full 162-run matrix.

Usage:
    python scripts/run_phase1_baselines.py
"""

import os
import sys
import time

# Allow running as a standalone script by putting the repo root on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.runner import run_experiment
from src.utils.config import load_config

BASELINE_CONFIGS = [
    "configs/core/pneumoniamnist_p0_ft_r100_s0.yaml",
    "configs/core/pneumoniamnist_p0_ft_r100_s1.yaml",
    "configs/core/pneumoniamnist_p0_ft_r100_s2.yaml",
    "configs/core/dermamnist_p0_ft_r100_s0.yaml",
    "configs/core/dermamnist_p0_ft_r100_s1.yaml",
    "configs/core/dermamnist_p0_ft_r100_s2.yaml",
]


def main() -> None:
    total = len(BASELINE_CONFIGS)
    all_results = []

    for idx, config_path in enumerate(BASELINE_CONFIGS, 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{total}] Running: {config_path}")
        print(f"{'='*70}")

        config = load_config(config_path)
        start = time.time()
        results = run_experiment(config)
        elapsed = time.time() - start

        all_results.append(results)
        print(f"\n--- Finished {config_path} in {elapsed:.1f}s ---")
        for key, value in results.items():
            print(f"  {key}: {value}")

    # Print summary table
    print(f"\n{'='*70}")
    print("PHASE 1 BASELINE SUMMARY")
    print(f"{'='*70}")
    print(f"{'Run':<40} {'AUROC':>8} {'Acc':>8} {'F1':>8} {'ECE':>8} {'Time':>8}")
    print("-" * 80)
    for r in all_results:
        print(
            f"{r['run']:<40} "
            f"{r['auroc']:>8.4f} "
            f"{r['accuracy']:>8.4f} "
            f"{r['macro_f1']:>8.4f} "
            f"{r['ece']:>8.4f} "
            f"{r['wall_clock_s']:>7.1f}s"
        )

    # Compute per-dataset mean ± std
    import numpy as np
    print(f"\n{'='*70}")
    print("PER-DATASET MEAN ± STD (3 seeds)")
    print(f"{'='*70}")
    for dataset in ["pneumoniamnist", "dermamnist"]:
        ds_results = [r for r in all_results if r["dataset"] == dataset]
        if not ds_results:
            continue
        aurocs = np.array([r["auroc"] for r in ds_results])
        accs = np.array([r["accuracy"] for r in ds_results])
        f1s = np.array([r["macro_f1"] for r in ds_results])
        eces = np.array([r["ece"] for r in ds_results])
        print(f"\n{dataset}:")
        print(f"  AUROC    = {aurocs.mean():.4f} ± {aurocs.std():.4f}")
        print(f"  Accuracy = {accs.mean():.4f} ± {accs.std():.4f}")
        print(f"  Macro-F1 = {f1s.mean():.4f} ± {f1s.std():.4f}")
        print(f"  ECE      = {eces.mean():.4f} ± {eces.std():.4f}")


if __name__ == "__main__":
    main()
