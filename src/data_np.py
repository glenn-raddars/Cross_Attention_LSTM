from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


FEATURE_COLUMNS = ["cn0", "elevation", "azimuth", "pseudorange_residual"]
REQUIRED_COLUMNS = ["location_id", "epoch", "satellite_id", "label", *FEATURE_COLUMNS]

ALIASES = {
    "cn0": ["cn0", "c_n0", "c/n0", "carrier_to_noise", "carrier-to-noise", "snr"],
    "elevation": ["elevation", "el", "elev"],
    "azimuth": ["azimuth", "az", "azi"],
    "pseudorange_residual": ["pseudorange_residual", "pre", "residual", "pseudorange_error"],
    "label": ["label", "is_los", "los", "target"],
    "epoch": ["epoch", "time", "timestamp", "tow"],
    "satellite_id": ["satellite_id", "sat_id", "svid", "prn"],
    "location_id": ["location_id", "loc", "site", "scene"],
}

Record = dict[str, str | float]


@dataclass(frozen=True)
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, records: list[Record]) -> "Normalizer":
        values = np.array([[float(row[column]) for column in FEATURE_COLUMNS] for row in records], dtype=np.float32)
        return cls(values.mean(axis=0), values.std(axis=0) + 1e-6)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values.astype(np.float32) - self.mean) / self.std


@dataclass(frozen=True)
class SplitRecords:
    train: list[Record]
    test: list[Record]


def _canonical_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def read_measurements(path: str | Path) -> list[Record]:
    reverse_aliases = {}
    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            reverse_aliases[_canonical_name(alias)] = canonical

    records: list[Record] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header.")
        column_map = {
            column: reverse_aliases[_canonical_name(column)]
            for column in reader.fieldnames
            if _canonical_name(column) in reverse_aliases
        }
        missing = [column for column in REQUIRED_COLUMNS if column not in set(column_map.values())]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        for raw in reader:
            row: Record = {}
            try:
                for original, canonical in column_map.items():
                    if canonical in FEATURE_COLUMNS or canonical in {"epoch", "label"}:
                        row[canonical] = float(raw[original])
                    else:
                        row[canonical] = str(raw[original])
            except (TypeError, ValueError):
                continue
            records.append(row)

    records.sort(key=lambda row: (str(row["location_id"]), str(row["satellite_id"]), float(row["epoch"])))
    return records


def _group(records: list[Record], keys: tuple[str, ...]) -> dict[tuple[object, ...], list[Record]]:
    groups: dict[tuple[object, ...], list[Record]] = {}
    for row in records:
        groups.setdefault(tuple(row[column] for column in keys), []).append(row)
    return groups


def split_in_domain(records: list[Record], train_ratio: float = 0.7, seed: int = 42) -> SplitRecords:
    rng = np.random.default_rng(seed)
    train: list[Record] = []
    test: list[Record] = []
    for group in _group(records, ("location_id",)).values():
        epochs = np.array(sorted({float(row["epoch"]) for row in group}))
        rng.shuffle(epochs)
        train_epochs = set(epochs[: max(1, int(len(epochs) * train_ratio))])
        train.extend(row for row in group if float(row["epoch"]) in train_epochs)
        test.extend(row for row in group if float(row["epoch"]) not in train_epochs)
    return SplitRecords(train, test)


def split_out_domain(records: list[Record], test_locations: Iterable[str] | None, seed: int = 42) -> SplitRecords:
    locations = np.array(sorted({str(row["location_id"]) for row in records}))
    if test_locations is None:
        rng = np.random.default_rng(seed)
        rng.shuffle(locations)
        count = max(1, int(round(len(locations) * 5 / 27)))
        test_locations = set(locations[:count])
    else:
        test_locations = {str(location) for location in test_locations}
    return SplitRecords(
        [row for row in records if str(row["location_id"]) not in test_locations],
        [row for row in records if str(row["location_id"]) in test_locations],
    )


def max_satellites(records: list[Record]) -> int:
    return max(len(group) for group in _group(records, ("location_id", "epoch")).values())


class GNSSNLOSDataset(Dataset):
    def __init__(
        self,
        records: list[Record],
        *,
        normalizer: Normalizer,
        window_size: int = 10,
        max_sats: int | None = None,
        min_history: int = 1,
    ) -> None:
        self.window_size = window_size
        self.max_sats = max_sats or max_satellites(records)
        self.normalizer = normalizer
        by_epoch = _group(records, ("location_id", "epoch"))
        by_track = _group(records, ("location_id", "satellite_id"))
        for group in by_epoch.values():
            group.sort(key=lambda row: str(row["satellite_id"]))
        for group in by_track.values():
            group.sort(key=lambda row: float(row["epoch"]))

        self.samples: list[dict[str, np.ndarray | float]] = []
        for track in by_track.values():
            for row_idx, row in enumerate(track):
                history = track[max(0, row_idx - window_size + 1) : row_idx + 1]
                if len(history) < min_history:
                    continue
                temporal = np.zeros((window_size, len(FEATURE_COLUMNS)), dtype=np.float32)
                hist_values = self._features(history)
                temporal[-len(history) :] = hist_values

                epoch_group = by_epoch[(row["location_id"], row["epoch"])]
                spatial_values = self._features(epoch_group)
                spatial = np.zeros((self.max_sats, len(FEATURE_COLUMNS)), dtype=np.float32)
                spatial_mask = np.ones((self.max_sats,), dtype=bool)
                keep = min(len(spatial_values), self.max_sats)
                spatial[:keep] = spatial_values[:keep]
                spatial_mask[:keep] = False

                target_position = 0
                for pos, epoch_row in enumerate(epoch_group[: self.max_sats]):
                    if epoch_row["satellite_id"] == row["satellite_id"]:
                        target_position = pos
                        break

                self.samples.append(
                    {
                        "temporal": temporal,
                        "spatial": spatial,
                        "spatial_mask": spatial_mask,
                        "instant": hist_values[-1],
                        "target_index": target_position,
                        "label": float(row["label"]),
                    }
                )

    def _features(self, rows: list[Record]) -> np.ndarray:
        values = np.array([[float(row[column]) for column in FEATURE_COLUMNS] for row in rows], dtype=np.float32)
        return self.normalizer.transform(values)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {
            "temporal": torch.tensor(sample["temporal"], dtype=torch.float32),
            "spatial": torch.tensor(sample["spatial"], dtype=torch.float32),
            "spatial_mask": torch.tensor(sample["spatial_mask"], dtype=torch.bool),
            "instant": torch.tensor(sample["instant"], dtype=torch.float32),
            "target_index": torch.tensor(sample["target_index"], dtype=torch.long),
            "label": torch.tensor(sample["label"], dtype=torch.float32),
        }
