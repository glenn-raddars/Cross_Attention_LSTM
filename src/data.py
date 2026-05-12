from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


FEATURE_COLUMNS = ["cn0", "elevation", "azimuth", "pseudorange_residual"] # 载噪比、仰角、方位角、伪距残差，这四个特征在 GNSS LOS/NLOS 分类任务中被广泛使用，且在大多数数据集中都能找到对应的列。我们把它们作为模型输入的核心特征，后续模型设计也围绕这四个特征展开。
# 数据集中必须具备的语义字段。后续模型默认每条观测都有位置、历元、卫星编号、
# 标签，以及 4 个用于分类的 GNSS 测量特征。
REQUIRED_COLUMNS = ["location_id", "epoch", "satellite_id", "label", *FEATURE_COLUMNS] # 位置、历元、卫星编号、LOS/NLOS 标签，以及模型输入的 4 个核心特征。这些字段在后续数据处理和模型训练中都是必不可少的，如果缺失会导致无法构造样本或训练模型。

# 不同数据源常会用不同列名表示同一个物理量，例如 cn0 也可能写成 snr。
# 这里把这些别名统一映射到代码内部使用的标准列名，降低 CSV 格式差异带来的影响。
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

Record = dict[str, str | float] # 定义一个 Record 类型，表示 CSV 中的一行数据。每个字段的值可以是字符串（例如 location_id、satellite_id）或者浮点数（例如 epoch、label 和模型输入特征）。在 read_measurements 函数中，我们会把 CSV 中的列名统一映射到标准列名，并且把数值字段转换成 float 类型，最终构造出一个 List[Record] 供后续处理使用。

# 让 Python 自动帮你生成一些常用方法，比如 __init__、__repr__ 等。 frozen=True 表示这个类的实例是不可变的，一旦创建就不能修改它的属性，这样可以保证在训练过程中不会意外改变均值和标准差。
@dataclass(frozen=True) # 装饰器的本质是定义完函数或类之后，把它传给装饰器函数处理，再把处理后的结果赋值回原来的名字。
class Normalizer:
    """保存训练集特征均值和标准差，并对输入特征做标准化。"""

    mean: np.ndarray
    std: np.ndarray

    @classmethod # @classmethod 表示这个方法是类方法，调用时不依赖某个已经创建好的对象，而是直接通过类来调用： Normalizer.fit(records) 就可以得到一个 Normalizer 实例。
    def fit(cls, records: list[Record]) -> "Normalizer": # 这里的cls参数代表 Normalizer 类本身，fit方法的作用是根据训练集的记录计算出每个特征的均值和标准差，并返回一个 Normalizer 实例，这个实例包含了这些统计信息，可以用来对输入特征进行标准化处理。
        # 只用训练集拟合均值/标准差，避免把测试集分布信息泄漏到训练过程。
        values = np.array([[float(row[column]) for column in FEATURE_COLUMNS] for row in records], dtype=np.float32) # values 的形状是 [样本数量, 特征数量]，每行对应一个样本的四个特征值。我们通过列表推导式遍历 records 中的每一行（每个 Record），按照 FEATURE_COLUMNS 中定义的顺序提取出 cn0、elevation、azimuth 和 pseudorange_residual 这四个特征，并把它们转换成 float 类型。最终得到一个二维的 NumPy 数组，供后续计算均值和标准差使用。
        # 加 1e-6 防止某个特征方差为 0 时除以 0。
        return cls(values.mean(axis=0), values.std(axis=0) + 1e-6) # 计算每个特征的均值和标准差，axis=0 表示按列计算，也就是对所有样本的同一特征计算均值和标准差。返回一个 Normalizer 实例，包含了每个特征的均值和标准差。相当于mean和std都是长度为4的一维数组，对应四个特征的统计信息。

    def transform(self, values: np.ndarray) -> np.ndarray:
        # 标准化后每个特征大致处在相近尺度，便于神经网络训练。
        return (values.astype(np.float32) - self.mean) / self.std


@dataclass(frozen=True) # 这个类用来保存训练集和测试集的记录列表，方便后续模型训练和评估使用。train 和 test 都是 List[Record] 类型，分别包含了训练集和测试集的记录。
class SplitRecords:
    train: list[Record]
    test: list[Record]


def _canonical_name(name: str) -> str:
    # 将 CSV 列名归一成统一格式，便于和 ALIASES 中的别名匹配。
    return name.strip().lower().replace(" ", "_").replace("-", "_") # strip() 去掉字符串两端的空白字符，lower() 转成小写，replace(" ", "_") 把空格替换成下划线，replace("-", "_") 把连字符替换成下划线。这样处理后，" C/N0 "、"c-n0"、"Carrier To Noise" 等不同格式的列名都能统一成 "cn0"，方便后续识别和映射。


def read_measurements(path: str | Path) -> list[Record]: # path 参数可以是字符串类型的文件路径，也可以是 Path 对象。这个函数的作用是读取 CSV 文件，统一列名、转换类型，并返回按轨迹时间排序后的记录列表，供后续数据处理和模型训练使用。
    """读取 CSV 文件，统一列名、转换类型，并返回按轨迹时间排序后的记录。"""

    # 反向索引：别名 -> 标准列名。例如 "snr" -> "cn0"。
    reverse_aliases = {}
    for canonical, aliases in ALIASES.items(): # 标准列名 canonical 和它的别名列表 aliases。我们遍历 ALIASES 中的每个标准列名和对应的别名列表，把每个别名都映射回标准列名，构建一个反向索引 reverse_aliases，这样在读取 CSV 时就能把各种不同的列名统一映射到代码内部使用的标准列名。
        for alias in aliases:
            reverse_aliases[_canonical_name(alias)] = canonical

    records: list[Record] = [] # 
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header.")
        # column_map 记录原始 CSV 列名到标准列名的映射，只保留代码认识的列。
        column_map = {
            column: reverse_aliases[_canonical_name(column)]
            for column in reader.fieldnames
            if _canonical_name(column) in reverse_aliases
        }
        # 确保训练所需字段全部能从 CSV 里找到，否则后面构造样本会缺关键信息。
        missing = [column for column in REQUIRED_COLUMNS if column not in set(column_map.values())]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        for raw in reader:
            row: Record = {}
            try:
                for original, canonical in column_map.items(): # original 是 CSV 中的列名，canonical 是代码内部使用的标准列名。我们遍历 column_map 中的每个原始列名和对应的标准列名，从 CSV 的每一行 raw 中提取出原始列名对应的值，并根据字段类型进行转换。
                    # epoch、label 和模型输入特征都参与数值计算，因此转成 float。
                    # location_id、satellite_id 用于分组和匹配，保留为字符串更稳妥。
                    if canonical in FEATURE_COLUMNS or canonical in {"epoch", "label"}:
                        row[canonical] = float(raw[original])
                    else:
                        row[canonical] = str(raw[original])
            except (TypeError, ValueError):
                # 遇到无法转换的坏行时直接跳过，避免单条脏数据中断整个训练流程。
                continue
            records.append(row)

    # 先按位置、卫星，再按时间排序，保证后续按卫星轨迹取历史窗口时顺序正确。
    records.sort(key=lambda row: (str(row["location_id"]), str(row["satellite_id"]), float(row["epoch"])))
    return records 
    # records的样例格式如下：
    # [
    #     {
    #         "location_id": "loc1",
    #         "epoch": 123456.0,
    #         "satellite_id": "sat1",
    #         "label": 1.0,
    #         "cn0": 45.0,
    #         "elevation": 30.0,
    #         "azimuth": 180.0,
    #         "pseudorange_residual": 0.5,
    #     },
    #     {
    #         "location_id": "loc1",
    #         "epoch": 123456.0,
    #         "satellite_id": "sat2",
    #         "label": 0.0,
    #         "cn0": 30.0,
    #         "elevation": 20.0,
    #         "azimuth": 190.0,
    #         "pseudorange_residual": 1.0,
    #     },
    #     ...
    # ]


def _group(records: list[Record], keys: tuple[str, ...]) -> dict[tuple[object, ...], list[Record]]:
    """按一个或多个字段分组，返回 key 元组到记录列表的映射。"""

    groups: dict[tuple[object, ...], list[Record]] = {}
    for row in records:
        groups.setdefault(tuple(row[column] for column in keys), []).append(row)
    return groups
    # groups 的格式如下：
    # {
    #     ("loc1",): [Record1, Record2, ...], # 按 location_id 分组，key 是一个包含 location_id 的元组，value 是这个 location_id 下的所有记录列表。
    #     ("loc2",): [Record3, Record4, ...],
    #     ...
    # }
    # 或者按 location_id 和 epoch 分组：
    # {
    #     ("loc1", 123456.0): [Record1, Record2, ...], # 按 location_id 和 epoch 分组，key 是一个包含 location_id 和 epoch 的元组，value 是这个 location_id 和 epoch 下的所有记录列表。
    #     ("loc1", 123457.0): [Record5, Record6, ...], 
    #     ("loc2", 123456.0): [Record3, Record4, ...],
    #     ...
    # }


def split_in_domain(records: list[Record], train_ratio: float = 0.7, seed: int = 42) -> SplitRecords:
    """同域划分：每个 location 内按 epoch 随机拆分训练集和测试集。"""

    rng = np.random.default_rng(seed)
    train: list[Record] = []
    test: list[Record] = []
    for group in _group(records, ("location_id",)).values():
        # 以 epoch 为单位划分，而不是逐行随机划分，避免同一历元的卫星集合被拆散后泄漏。
        epochs = np.array(sorted({float(row["epoch"]) for row in group}))
        rng.shuffle(epochs)
        train_epochs = set(epochs[: max(1, int(len(epochs) * train_ratio))])
        train.extend(row for row in group if float(row["epoch"]) in train_epochs)
        test.extend(row for row in group if float(row["epoch"]) not in train_epochs)
    return SplitRecords(train, test)


def split_out_domain(records: list[Record], test_locations: Iterable[str] | None, seed: int = 42) -> SplitRecords:
    """跨域划分：用部分 location 作为测试场景，其余 location 作为训练场景。"""

    locations = np.array(sorted({str(row["location_id"]) for row in records}))
    if test_locations is None:
        rng = np.random.default_rng(seed)
        rng.shuffle(locations)
        # 论文数据常以 27 个场景、约 5 个测试场景做 out-domain 设置；这里按比例泛化。
        count = max(1, int(round(len(locations) * 5 / 27)))
        test_locations = set(locations[:count])
    else:
        # 命令行传入的 location 统一转成字符串，和 read_measurements 中的类型保持一致。
        test_locations = {str(location) for location in test_locations}
    return SplitRecords(
        [row for row in records if str(row["location_id"]) not in test_locations],
        [row for row in records if str(row["location_id"]) in test_locations],
    )


def max_satellites(records: list[Record]) -> int:
    # 找到任意 location+epoch 下最多同时出现多少颗卫星，用作 spatial 序列的 padding 长度。
    return max(len(group) for group in _group(records, ("location_id", "epoch")).values())


class GNSSNLOSDataset(Dataset):
    """把原始记录转换为模型训练所需的 PyTorch Dataset。

    每个样本对应某一位置、某一历元、某一颗目标卫星，包含：
    temporal: 目标卫星最近 window_size 个历元的历史特征。
    spatial: 当前历元同一位置下所有卫星的特征集合。
    spatial_mask: 标记 spatial 中哪些位置是 padding，供注意力模块忽略。
    instant: 目标卫星当前时刻特征，供 MLP baseline 使用。
    target_index: 目标卫星在 spatial 序列中的位置，供 TBM 取回目标卫星表示。
    label: 当前目标卫星的 LOS/NLOS 标签。
    """

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
        self.max_sats = max_sats or max_satellites(records) # 最大卫星数如果不指定就自动计算，保证空间输入能容纳每个历元的所有卫星。这个值会用来确定 spatial 输入的第二维长度，以及 spatial_mask 的长度。
        self.normalizer = normalizer
        # by_epoch 用于构造同一历元的空间卫星集合；by_track 用于构造同一卫星的时间历史。
        by_epoch = _group(records, ("location_id", "epoch"))
        by_track = _group(records, ("location_id", "satellite_id"))
        for group in by_epoch.values():
            # 同一历元内按 satellite_id 排序，保证 spatial 序列顺序稳定。
            group.sort(key=lambda row: str(row["satellite_id"]))
        for group in by_track.values():
            # 同一卫星轨迹按 epoch 排序，保证 temporal 历史从早到晚排列。
            group.sort(key=lambda row: float(row["epoch"]))

        self.samples: list[dict[str, np.ndarray | float]] = []
        for track in by_track.values():
            for row_idx, row in enumerate(track):
                # 取当前行及其之前的 window_size-1 条观测，形成目标卫星的历史窗口。
                history = track[max(0, row_idx - window_size + 1) : row_idx + 1]
                if len(history) < min_history:
                    continue
                # temporal 固定为 [window_size, feature_dim]。历史不足 window_size 时，
                # 前面保持 0 padding，真实历史右对齐放在末尾，最近时刻始终在最后一行。
                temporal = np.zeros((window_size, len(FEATURE_COLUMNS)), dtype=np.float32)
                hist_values = self._features(history) 
                temporal[-len(history) :] = hist_values # temporal 的形状是 [window_size, feature_dim]，其中 feature_dim 是 FEATURE_COLUMNS 的长度，也就是 4。对于每个样本，我们取当前行所在卫星轨迹的历史窗口，如果历史长度不足 window_size，就在前面用零填充，保证 temporal 的形状固定。最近的历史时刻始终放在 temporal 的最后一行，这样模型就能学会关注最近的历史信息。

                # 当前历元下所有卫星共同构成空间输入，供 AAM/Transformer 建模卫星间关系。
                epoch_group = by_epoch[(row["location_id"], row["epoch"])]
                # epoch_group的样例格式如下：
                # [
                #     {
                #         "location_id": "loc1",
                #         "epoch": 123456.0,
                #         "satellite_id": "sat1",
                #         "label": 1.0,
                #         "cn0": 45.0,
                #         "elevation": 30.0,
                #         "azimuth": 180.0,
                #         "pseudorange_residual": 0.5,
                #     },
                #     {
                #         "location_id": "loc1",
                #         "epoch": 123456.0,
                #         "satellite_id": "sat2",
                #         "label": 0.0,
                #         "cn0": 30.0,
                #         "elevation": 20.0,
                #         "azimuth": 190.0,
                #         "pseudorange_residual": 1.0,
                #     },
                #     ...
                # ]
                spatial_values = self._features(epoch_group)
                spatial = np.zeros((self.max_sats, len(FEATURE_COLUMNS)), dtype=np.float32)
                # PyTorch MultiheadAttention 的 key_padding_mask 中 True 表示“需要忽略”。
                # 因此先全部置 True，再把真实卫星位置改成 False。
                spatial_mask = np.ones((self.max_sats,), dtype=bool)
                keep = min(len(spatial_values), self.max_sats)
                spatial[:keep] = spatial_values[:keep] # 把有用的卫星特征填充到spatial的前 keep 行，剩余行保持全零。spatial 的形状是 [max_sats, feature_dim]，其中 max_sats 是我们之前确定的最大卫星数，feature_dim 是 FEATURE_COLUMNS 的长度。对于每个样本，我们取当前历元下所有卫星的特征，如果卫星数量超过 max_sats，就只保留前 max_sats 个；如果不足，就在后面用零填充。
                spatial_mask[:keep] = False

                # 记录目标卫星在 spatial 序列中的位置。TransformerBasedModel 会用这个
                # index 从空间编码结果 hs: [B, N, hidden_dim] 中取出目标卫星的表示。
                target_position = 0
                for pos, epoch_row in enumerate(epoch_group[: self.max_sats]):
                    if epoch_row["satellite_id"] == row["satellite_id"]:
                        target_position = pos
                        break

                self.samples.append(
                    {
                        # proposed/fusion/concate 模型使用 temporal 和 spatial。
                        "temporal": temporal,
                        "spatial": spatial,
                        "spatial_mask": spatial_mask,
                        # MLPBaseline 只使用当前目标卫星的 instant 特征。
                        "instant": hist_values[-1],
                        "target_index": target_position,
                        "label": float(row["label"]),
                    }
                )

    def _features(self, rows: list[Record]) -> np.ndarray:
        # 按 FEATURE_COLUMNS 固定顺序抽取特征，保证训练和推理时特征维度含义一致。
        values = np.array([[float(row[column]) for column in FEATURE_COLUMNS] for row in rows], dtype=np.float32)
        return self.normalizer.transform(values) # 这里调用的是类方法 Normalizer.transform，对输入特征进行标准化处理，保证每个特征大致处在相近尺度，便于神经网络训练。

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        # DataLoader 会把这些单样本 tensor 自动 stack 成 batch：
        # temporal -> [B, T, F]，spatial -> [B, N, F]，spatial_mask -> [B, N]。
        return {
            "temporal": torch.tensor(sample["temporal"], dtype=torch.float32),
            "spatial": torch.tensor(sample["spatial"], dtype=torch.float32),
            "spatial_mask": torch.tensor(sample["spatial_mask"], dtype=torch.bool),
            "instant": torch.tensor(sample["instant"], dtype=torch.float32),
            "target_index": torch.tensor(sample["target_index"], dtype=torch.long),
            "label": torch.tensor(sample["label"], dtype=torch.float32),
        }
