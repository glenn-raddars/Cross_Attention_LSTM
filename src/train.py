from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_np import FEATURE_COLUMNS, GNSSNLOSDataset, Normalizer, max_satellites, read_measurements, split_in_domain, split_out_domain
from models import build_model


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Metrics:
    model.eval()
    labels: list[int] = []
    preds: list[int] = []
    for batch in loader:
        batch = move_batch(batch, device)
        logits = model(batch)
        probabilities = torch.sigmoid(logits)
        preds.extend((probabilities >= 0.5).long().cpu().tolist())
        labels.extend(batch["label"].long().cpu().tolist())
    return Metrics(
        accuracy=accuracy_score(labels, preds),
        precision=precision_score(labels, preds, zero_division=0),
        recall=recall_score(labels, preds, zero_division=0),
        f1=f1_score(labels, preds, zero_division=0),
    )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    frame = read_measurements(args.data)
    if args.split == "in_domain":
        split = split_in_domain(frame, train_ratio=args.train_ratio, seed=args.seed)
    else:
        test_locations = args.test_locations.split(",") if args.test_locations else None
        split = split_out_domain(frame, test_locations=test_locations, seed=args.seed)

    normalizer = Normalizer.fit(split.train)
    max_sats = max_satellites(frame)
    train_set = GNSSNLOSDataset(split.train, normalizer=normalizer, window_size=args.window_size, max_sats=max_sats)
    test_set = GNSSNLOSDataset(split.test, normalizer=normalizer, window_size=args.window_size, max_sats=max_sats)
    if len(train_set) == 0 or len(test_set) == 0:
        raise ValueError("The selected split produced an empty train or test dataset.")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(
        args.model,
        input_dim=len(FEATURE_COLUMNS),
        hidden_dim=args.hidden_dim,
        ff_dim=args.ff_dim,
        heads=args.heads,
        aam_layers=args.aam_layers,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    criterion = nn.BCEWithLogitsLoss()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for batch in bar:
            batch = move_batch(batch, device)
            logits = model(batch)
            loss = criterion(logits, batch["label"])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.item()))
            bar.set_postfix(loss=np.mean(losses))

        metrics = evaluate(model, test_loader, device)
        record = {"epoch": epoch, "loss": float(np.mean(losses)), **asdict(metrics)}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if metrics.f1 > best_f1:
            best_f1 = metrics.f1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "normalizer": {"mean": normalizer.mean.tolist(), "std": normalizer.std.tolist()},
                    "metrics": asdict(metrics),
                },
                args.output_dir / f"{args.model}_{args.split}_best.pt",
            )

    with (args.output_dir / f"{args.model}_{args.split}_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PyTorch GNSS NLOS reproduction.")
    parser.add_argument("--data", type=Path, required=True, help="CSV with GNSS measurements.")
    parser.add_argument("--split", choices=["in_domain", "out_domain"], default="in_domain")
    parser.add_argument("--test-locations", default="", help="Comma separated location IDs for out-domain testing.")
    parser.add_argument("--model", choices=["proposed", "fusion", "concate", "mlp", "tbm", "fcnn_lstm"], default="proposed")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--aam-layers", type=int, default=1)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
