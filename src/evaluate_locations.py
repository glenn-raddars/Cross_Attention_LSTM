from __future__ import annotations

"""按 location 评估已训练模型的脚本。

用途：加载一个训练好的 checkpoint（通常是 out_domain 训练得到的），
在指定的若干 location（如 P2、P4）上分别评估分类表现，用来观察模型
在不同场景下的迁移/泛化能力。

与 train.py 的关键区别：
1. 不重新训练，只做前向推理。
2. 复用 checkpoint 里保存的 normalizer（训练集均值/方差），保证输入
   分布和训练时一致，绝不在评估数据上重新 fit。
3. 逐 location 分别评估，并标注每个 location 相对该 checkpoint 的角色
   （训练见过 / out-domain 测试 / 完全没见过），避免把 in-domain 结果
   误读成泛化能力。

示例：
    python src/evaluate_locations.py \
        --checkpoint runs/proposed_model_outdomain/proposed_out_domain_best.pt \
        --data data/los_nlos_static.csv \
        --locations P2,P4
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader

from data import (
    FEATURE_COLUMNS,
    GNSSNLOSDataset,
    Normalizer,
    max_satellites,
    read_measurements,
)
from models import build_model


@dataclass
class LocationResult:
    """单个 location（或聚合）上的评估结果。"""

    location: str
    role: str            # train(seen) / out-domain / unseen / n/a
    n_samples: int
    pos_rate: float      # 正类（label=1）样本占比，反映类别是否均衡
    accuracy: float
    precision: float
    recall: float
    f1: float


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int]]:
    """在给定 loader 上前向推理，返回 (labels, preds) 两个 0/1 列表。"""

    model.eval()
    labels: list[int] = []
    preds: list[int] = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(batch)
        probabilities = torch.sigmoid(logits)
        preds.extend((probabilities >= 0.5).long().cpu().tolist())
        labels.extend(batch["label"].long().cpu().tolist())
    return labels, preds


def _evaluate_records(
    records: list,
    *,
    location_name: str,
    role: str,
    model: nn.Module,
    normalizer: Normalizer,
    window_size: int,
    max_sats: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> LocationResult | None:
    """把一批记录构造成 Dataset 并评估；样本为空时返回 None。"""

    dataset = GNSSNLOSDataset(
        records,
        normalizer=normalizer,
        window_size=window_size,
        max_sats=max_sats,
    )
    if len(dataset) == 0:
        return None

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    labels, preds = _predict(model, loader, device)
    return LocationResult(
        location=location_name,
        role=role,
        n_samples=len(labels),
        pos_rate=float(np.mean(labels)) if labels else 0.0,
        accuracy=accuracy_score(labels, preds),
        precision=precision_score(labels, preds, zero_division=0),
        recall=recall_score(labels, preds, zero_division=0),
        f1=f1_score(labels, preds, zero_division=0),
    )


def _print_table(results: list[LocationResult]) -> None:
    """把结果按固定宽度对齐打印成一张表。"""

    header = f"{'location':<12}{'role':<14}{'n_samples':>10}{'pos_rate':>10}{'accuracy':>10}{'precision':>11}{'recall':>9}{'f1':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.location:<12}{r.role:<14}{r.n_samples:>10}{r.pos_rate:>10.3f}"
            f"{r.accuracy:>10.4f}{r.precision:>11.4f}{r.recall:>9.4f}{r.f1:>9.4f}"
        )


def evaluate(args: argparse.Namespace) -> None:
    """评估入口：加载 checkpoint，按 location 评估并打印结果。"""

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # 1) 加载 checkpoint。weights_only=False 因为里面还存了 args/normalizer 等非张量信息。
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = ckpt["args"]

    # 2) 用 checkpoint 里记录的超参数复原模型结构，再加载训练好的权重。
    model = build_model(
        train_args["model"],
        input_dim=len(FEATURE_COLUMNS),
        hidden_dim=train_args["hidden_dim"],
        ff_dim=train_args["ff_dim"],
        heads=train_args["heads"],
        aam_layers=train_args["aam_layers"],
        lstm_layers=train_args["lstm_layers"],
        dropout=train_args["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    # 3) 复用训练集拟合出来的 normalizer（关键：评估必须沿用同一套均值/方差）。
    normalizer = Normalizer(
        mean=np.array(ckpt["normalizer"]["mean"], dtype=np.float32),
        std=np.array(ckpt["normalizer"]["std"], dtype=np.float32),
    )

    window_size = int(train_args["window_size"])

    # 4) 读入完整数据，并按训练时同样的方式计算 max_sats（spatial padding 长度）。
    frame = read_measurements(args.data)
    max_sats = max_satellites(frame)

    # 5) 还原该 checkpoint 的训练/测试 location 划分，用于给每个 location 标注角色。
    test_locations = {
        loc.strip() for loc in str(train_args.get("test_locations", "") or "").split(",") if loc.strip()
    }
    all_locations = sorted({str(row["location_id"]) for row in frame})
    is_out_domain_ckpt = train_args.get("split") == "out_domain"

    def role_of(location: str) -> str:
        if location not in all_locations:
            return "unseen"          # 数据里根本没有这个 location
        if not is_out_domain_ckpt:
            return "n/a"             # in_domain checkpoint 无法干净地区分训练/测试 location
        if location in test_locations:
            return "out-domain"      # 训练时留出的测试场景，真正的泛化指标
        return "train(seen)"         # 训练见过的场景，指标偏乐观

    # 6) 确定要评估哪些 location：未指定时默认评估该 checkpoint 原本的 out-domain 测试集。
    if args.locations:
        requested = [loc.strip() for loc in args.locations.split(",") if loc.strip()]
    elif test_locations:
        requested = sorted(test_locations)
        print(f"未指定 --locations，默认评估 checkpoint 的 out-domain 测试集: {', '.join(requested)}")
    else:
        requested = all_locations
        print(f"未指定 --locations 且 checkpoint 无 test_locations，评估全部 location: {', '.join(requested)}")

    print(
        f"\nCheckpoint: {args.checkpoint}\n"
        f"Model: {train_args['model']} | window={window_size} | "
        f"训练时 test_locations: {', '.join(sorted(test_locations)) or '(无)'}\n"
        f"数据中全部 location: {', '.join(all_locations)}\n"
    )

    # 7) 逐 location 评估。
    results: list[LocationResult] = []
    requested_present: list[str] = []
    for location in requested:
        rows = [row for row in frame if str(row["location_id"]) == location]
        if not rows:
            print(f"[warn] location {location} 在数据中不存在，跳过。")
            continue
        result = _evaluate_records(
            rows,
            location_name=location,
            role=role_of(location),
            model=model,
            normalizer=normalizer,
            window_size=window_size,
            max_sats=max_sats,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        if result is None:
            print(f"[warn] location {location} 构造不出样本（可能历元过少），跳过。")
            continue
        results.append(result)
        requested_present.append(location)

    if not results:
        print("没有可评估的 location。")
        return

    # 8) 再算一个所有被评估 location 合在一起的聚合结果，便于和逐 location 对照。
    if len(requested_present) > 1:
        combined_rows = [row for row in frame if str(row["location_id"]) in set(requested_present)]
        combined = _evaluate_records(
            combined_rows,
            location_name="OVERALL",
            role="(combined)",
            model=model,
            normalizer=normalizer,
            window_size=window_size,
            max_sats=max_sats,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        if combined is not None:
            results.append(combined)

    print("Per-location results:")
    _print_table(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按 location 评估已训练的 GNSS NLOS 模型。")
    parser.add_argument("--checkpoint", type=Path, required=True, help="训练保存的 *_best.pt 路径。")
    parser.add_argument("--data", type=Path, required=True, help="包含全部 location 的 CSV 数据。")
    parser.add_argument(
        "--locations",
        default="",
        help="逗号分隔的 location ID（如 P2,P4）。不填则默认评估 checkpoint 的 out-domain 测试集。",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
