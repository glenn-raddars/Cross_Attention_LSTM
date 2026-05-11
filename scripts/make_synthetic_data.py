from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def make_synthetic(path: Path, locations: int, epochs: int, sats: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for loc in range(locations):
        canyon_bias = rng.normal(0.0, 0.8)
        blocked_azimuth = rng.uniform(0, 360)
        for epoch in range(epochs):
            temporal_phase = np.sin(epoch / 9.0 + loc)
            for sat in range(sats):
                azimuth = (sat * 360 / sats + epoch * 1.8 + rng.normal(0, 4)) % 360
                elevation = np.clip(rng.normal(42 + 15 * np.sin(epoch / 17 + sat), 18), 5, 85)
                angular_gap = min(abs(azimuth - blocked_azimuth), 360 - abs(azimuth - blocked_azimuth))
                blockage = angular_gap < (45 + 8 * canyon_bias)
                nlos_score = 1.4 * blockage + 0.8 * (elevation < 28) + 0.25 * temporal_phase + rng.normal(0, 0.45)
                label = 0 if nlos_score > 0.9 else 1
                cn0 = rng.normal(31 if label == 0 else 39, 3.5)
                residual = rng.normal(5.5 if label == 0 else 0.5, 2.0)
                rows.append(
                    {
                        "location_id": f"L{loc:02d}",
                        "epoch": epoch,
                        "satellite_id": f"S{sat:02d}",
                        "label": label,
                        "cn0": cn0,
                        "elevation": elevation,
                        "azimuth": azimuth,
                        "pseudorange_residual": residual,
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small paper-shaped GNSS NLOS CSV for smoke tests.")
    parser.add_argument("--out", type=Path, default=Path("data/synthetic_gnss.csv"))
    parser.add_argument("--locations", type=int, default=27)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--sats", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    make_synthetic(args.out, args.locations, args.epochs, args.sats, args.seed)
    print(args.out)


if __name__ == "__main__":
    main()
