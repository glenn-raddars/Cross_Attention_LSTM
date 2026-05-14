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

from data import FEATURE_COLUMNS, GNSSNLOSDataset, Normalizer, max_satellites, read_measurements, split_in_domain, split_out_domain
from models import build_model


@dataclass # 只修饰它紧随其后的类或者函数
class Metrics:
    # 统一保存一次评估得到的分类指标，后面可以很方便地转成 dict 写日志或保存 checkpoint。
    accuracy: float
    precision: float
    recall: float
    f1: float


def set_seed(seed: int) -> None:
    """固定各个随机源的种子，尽量保证数据划分、初始化和训练过程可复现。"""

    # Python、NumPy 和 PyTorch 都有各自的随机数生成器，需要分别设置。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 如果使用 CUDA，多卡/单卡上的 CUDA 随机状态也一并固定。
    torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """把一个 batch 中的所有张量搬到指定设备（CPU 或 GPU）。"""

    # Dataset 返回的是一个字典，模型会按 key 读取 temporal、spatial、label 等字段。
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Metrics:
    """在验证/测试集上前向推理，并计算二分类指标。"""

    # eval() 会关闭 dropout 等训练态行为，使评估结果更稳定。
    model.eval() # Dropout 在训练时通常不会把模型参数置零，它置零的是某一层的激活值 / 中间输出，也就是本次前向传播里神经元输出的部分元素。
    labels: list[int] = []
    preds: list[int] = []
    for batch in loader:
        batch = move_batch(batch, device)
        # 模型输出 logits，BCEWithLogitsLoss 训练时也直接吃 logits。
        logits = model(batch)
        # 评估时用 sigmoid 把 logits 转成 [0, 1] 概率，再用 0.5 作为二分类阈值。
        probabilities = torch.sigmoid(logits)
        preds.extend((probabilities >= 0.5).long().cpu().tolist()) # 先比较概率和 0.5 得到布尔值，再用 long() 转成 0/1 的整数标签，最后搬回 CPU 转成列表。extend的作用是把一个可迭代对象中的元素逐个添加到列表末尾，而不是把整个可迭代对象作为一个元素添加到列表中。
        labels.extend(batch["label"].long().cpu().tolist()) # batch["label"] 是原始标签张量，直接转成 long() 以确保是整数类型，再搬回 CPU 转成列表。
    # zero_division=0 避免某一类完全没有预测时 precision/recall 抛出警告或异常。
    return Metrics(
        accuracy=accuracy_score(labels, preds), # 准确率：预测正确的样本数占总样本数的比例。
        precision=precision_score(labels, preds, zero_division=0), # 精确率：预测为正类的样本中实际为正类的比例。
        recall=recall_score(labels, preds, zero_division=0), # 召回率：实际为正类的样本中被预测为正类的比例。
        f1=f1_score(labels, preds, zero_division=0), # F1 分数：精确率和召回率的调和平均数。
    )


def train(args: argparse.Namespace) -> None:
    """完整训练入口：读取数据、构造数据集、训练模型、评估并保存最佳权重。"""

    # 先固定随机性，再做数据划分和模型初始化，保证同一组参数下结果尽量一致。
    set_seed(args.seed)
    # 如果命令行没有指定 --device，就优先使用 CUDA；没有 GPU 时回退到 CPU。
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # 读取并清洗 CSV，返回已经标准化列名、转换数值类型、按轨迹排序的记录。
    frame = read_measurements(args.data)
    if args.split == "in_domain":
        # 同域划分：每个场景内部按 epoch 拆训练集/测试集，训练和测试来自相同 location 分布。
        split = split_in_domain(frame, train_ratio=args.train_ratio, seed=args.seed)
    else:
        # 跨域划分：指定或随机选择若干 location 作为测试场景，检验模型迁移能力。
        test_locations = args.test_locations.split(",") if args.test_locations else None
        split = split_out_domain(frame, test_locations=test_locations, seed=args.seed)

    # 只用训练集拟合标准化参数，避免测试集统计信息泄漏到训练过程。
    normalizer = Normalizer.fit(split.train)
    # 统计全数据中同一时刻最多出现的卫星数，用于把 spatial 输入 padding 到固定长度。
    max_sats = max_satellites(frame)
    # Dataset 会把原始记录转换成模型需要的 temporal/spatial/instant/label 等张量。
    train_set = GNSSNLOSDataset(split.train, normalizer=normalizer, window_size=args.window_size, max_sats=max_sats)
    test_set = GNSSNLOSDataset(split.test, normalizer=normalizer, window_size=args.window_size, max_sats=max_sats)
    if len(train_set) == 0 or len(test_set) == 0:
        raise ValueError("The selected split produced an empty train or test dataset.")

    # 训练集开启 shuffle 打乱样本顺序；测试集保持固定顺序，便于复现实验结果。
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 根据 --model 选择具体网络结构，并把共享超参数传入模型构造函数。
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
    # Adam 的 beta=(0.9, 0.98) 常用于 Transformer/注意力模型训练，收敛较平稳。
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    # 二分类任务使用 BCEWithLogitsLoss：内部会做 sigmoid，比手动 sigmoid 后再 BCE 更数值稳定。
    criterion = nn.BCEWithLogitsLoss()

    # 创建输出目录，用于保存最佳模型和每轮训练历史。
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        # train() 会启用 dropout、BatchNorm 等训练态行为。
        model.train()
        losses = []
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False) # leave=False 让 tqdm 在每轮结束后清除进度条，保持输出整洁。
        for batch in bar:
            batch = move_batch(batch, device)
            # 前向传播得到每个样本的 NLOS/LOS 二分类 logit。
            logits = model(batch)
            # label 由 Dataset 提供，形状应与 logits 对齐。
            loss = criterion(logits, batch["label"])
            # 清空上一轮梯度，反向传播当前 batch 的 loss。
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪可以缓解 LSTM/注意力模型训练中的梯度爆炸问题。
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.item()))
            # 在进度条上显示当前 epoch 的平均 loss，便于观察训练是否正常下降。
            bar.set_postfix(loss=np.mean(losses))

        # 每个 epoch 结束后在测试集上评估一次，并记录 loss 与分类指标。
        metrics = evaluate(model, test_loader, device)
        record = {"epoch": epoch, "loss": float(np.mean(losses)), **asdict(metrics)} # **asdict(metrics) 会把 Metrics 实例的字段展开成字典项，方便后续 JSON 序列化。
        history.append(record)
        # 按 JSON 行打印，方便后续被脚本或日志系统解析。
        print(json.dumps(record, ensure_ascii=False)) # ensure_ascii=False 让 JSON 输出中的非 ASCII 字符（如中文）直接显示，而不是被转义成 \uXXXX 的形式。
        if metrics.f1 > best_f1:
            # 以 F1 作为模型选择指标，适合 LOS/NLOS 类别可能不均衡的情况。
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

    # 保存完整训练历史，便于画曲线、比较不同模型或复现实验。
    with (args.output_dir / f"{args.model}_{args.split}_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, ensure_ascii=False) # indent=2 让 JSON 输出更易读，ensure_ascii=False 让非 ASCII 字符直接显示。


def parse_args() -> argparse.Namespace:
    """定义并解析命令行参数，所有训练超参数都从这里进入 train()。"""

    parser = argparse.ArgumentParser(description="Train the PyTorch GNSS NLOS reproduction.")
    # 数据与实验划分相关参数。
    parser.add_argument("--data", type=Path, required=True, help="CSV with GNSS measurements.")
    parser.add_argument("--split", choices=["in_domain", "out_domain"], default="in_domain")
    parser.add_argument("--test-locations", default="", help="Comma separated location IDs for out-domain testing.")
    # 模型结构与输出路径。
    parser.add_argument("--model", choices=["proposed", "fusion", "concate", "mlp", "tbm", "fcnn_lstm"], default="proposed")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    # 数据窗口和训练过程超参数。
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    # 网络宽度、注意力层数、LSTM 层数和正则化配置。
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--aam-layers", type=int, default=1)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    # 运行环境与复现相关参数。
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="")
    return parser.parse_args()


if __name__ == "__main__":
    # 直接运行 python src/train.py 时，从命令行读取参数并启动训练。
    train(parse_args())
