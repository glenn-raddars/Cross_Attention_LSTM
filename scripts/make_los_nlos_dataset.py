from __future__ import annotations

"""把 LOS-NLOS-main 的 7 个静态 GNSS 数据集（P2.csv ~ P8.csv）合并成本项目训练脚本
需要的单张扁平观测表。

背景
----
`data/LOS-NLOS-main/data for sharing_csv/` 下是另一套 STCA 交叉注意力 + LSTM 实现使用的
原始静态数据，每个测站一个 CSV，列结构为：

    GPS_Time(s), PRN, nSV, pseudorange, C/N0, Elevation, Azimuth, err_tropo, err_iono,
    sat_clock_error, Pseudorange_residual, Normalized_Pseudorange_residual,
    Pr_rate_consitency, Sat_pos_x, Sat_pos_y, Sat_pos_z, LOS/NLOS_label

而本项目（src/data.py、src/train.py）的入口 read_measurements() 只认识一张固定 8 列的
扁平表：

    location_id, epoch, satellite_id, label, cn0, elevation, azimuth, pseudorange_residual

并且原始表头里的 `GPS_Time(s)`、`LOS/NLOS_label` 不在 src/data.py 的列名别名表里，原始数据
也没有 location 列，因此不能直接喂给训练脚本，必须先转换。

转换规则（对照 LOS_NLOS_SPLIT_ANALYSIS.md 中对原项目预处理流程的分析）
------------------------------------------------------------------
1. 加载合并并添加 location（loaders.py:30-38、78-85）
   只发现文件名前缀在 P2~P8 中的 CSV；合并所有行，并给每行添加 location = 文件名。
   这样输出的 location_id 正好是 "P2".."P8"，配合 src/train.py 的
   `--split out_domain --test-locations P6,P7` 就能复现原项目固定的跨域划分。

2. 过滤异常值（filters.py:61-81）
   - 删除关键字段无法解析（NaN）的行；
   - 删除 Pr_rate_consitency == 9999.0 的行（该列在历元/卫星首次出现、无法计算变化率
     一致性时取 9999.0 作为哨兵值）；
   - 删除 abs(Pseudorange_residual) >= 100 的行（伪距残差野值）。

3. 标签映射（constants.py:14-16）
   LOS/NLOS_label 的 `-1 -> 0`（NLOS）、`1 -> 1`（LOS）。本项目模型输出 sigmoid 概率并用
   BCEWithLogitsLoss 训练，因此标签必须是 {0, 1}。

4. 选取 4 个输入特征（LOS_NLOS_SPLIT_ANALYSIS.md 对齐建议 #2）
   C/N0 -> cn0，Elevation -> elevation，Azimuth -> azimuth，
   Pseudorange_residual -> pseudorange_residual。

注意：原项目的“窗口化 / spatial padding”等步骤由其模型代码完成；本项目的
GNSSNLOSDataset（src/data.py）会自己按 (location_id, satellite_id) 和 (location_id, epoch)
重新构造 temporal / spatial 两路输入，所以这里只需输出干净的逐行观测表，不做窗口化。

用法
----
    python scripts/make_los_nlos_dataset.py
    # 然后即可训练，例如复现跨域划分：
    python src/train.py --data data/los_nlos_static.csv --split out_domain --test-locations P6,P7
    # 或同域划分：
    python src/train.py --data data/los_nlos_static.csv --split in_domain
"""

import argparse
import csv
import math
from pathlib import Path


# 只发现文件名前缀在该集合中的 CSV（对应 loaders.py:30-38 的 P2..P8 白名单）。
SOURCE_LOCATIONS = ["P2", "P3", "P4", "P5", "P6", "P7", "P8"]

# 原始 CSV 中要用到的列名（保持与源文件表头完全一致，注意 "Pr_rate_consitency" 是源文件
# 自带的拼写，不是笔误）。
SRC_TIME = "GPS_Time(s)"
SRC_PRN = "PRN"
SRC_CN0 = "C/N0"
SRC_ELEVATION = "Elevation"
SRC_AZIMUTH = "Azimuth"
SRC_RESIDUAL = "Pseudorange_residual"
SRC_RATE_CONSISTENCY = "Pr_rate_consitency"
SRC_LABEL = "LOS/NLOS_label"

# 输出列必须与 src/data.py / src/data_np.py 的 REQUIRED_COLUMNS、FEATURE_COLUMNS 对齐。
OUTPUT_COLUMNS = [
    "location_id",
    "epoch",
    "satellite_id",
    "label",
    "cn0",
    "elevation",
    "azimuth",
    "pseudorange_residual",
]

# 原始标签到本项目标签的映射：-1(NLOS) -> 0，1(LOS) -> 1。
LABEL_MAP = {-1: 0, 1: 1}


class FilterStats:
    """统计转换过程中各类被丢弃的行数，便于在结束时打印一份可核对的摘要。"""

    def __init__(self) -> None:
        self.read = 0            # 读入的原始行数
        self.bad_value = 0       # 关键字段无法解析为数值（NaN/脏行）
        self.rate_sentinel = 0   # Pr_rate_consitency 命中哨兵值被丢弃
        self.large_residual = 0  # abs(Pseudorange_residual) 超阈值被丢弃
        self.bad_label = 0       # 标签不是 -1 / 1
        self.kept = 0            # 最终保留的行数


def _parse_float(text: str) -> float:
    """把可能带前后空格的字符串解析成 float；空串/非数值会抛 ValueError。

    源 CSV 的数值列普遍带有对齐空格（例如 "  2"、" 50418"），float() 本身能处理前后
    空白，这里再显式判一次 NaN，避免 "nan" 字面量混进训练数据。
    """

    value = float(text)
    if math.isnan(value):
        raise ValueError("NaN")
    return value


def convert_file(
    path: Path,
    location_id: str,
    *,
    rate_sentinel: float,
    residual_limit: float,
    stats: FilterStats,
) -> list[dict[str, object]]:
    """把单个测站 CSV 转换成输出行列表，并就地累加过滤统计。"""

    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header.")
        missing = [
            column
            for column in (SRC_TIME, SRC_PRN, SRC_CN0, SRC_ELEVATION, SRC_AZIMUTH, SRC_RESIDUAL, SRC_RATE_CONSISTENCY, SRC_LABEL)
            if column not in reader.fieldnames
        ]
        if missing:
            raise ValueError(f"{path} is missing expected columns: {missing}")

        for raw in reader:
            stats.read += 1

            # --- 解析关键字段，任一无法解析就当作 NaN 行丢弃（filters.py 的 dropna）---
            try:
                epoch = _parse_float(raw[SRC_TIME])
                prn = int(round(_parse_float(raw[SRC_PRN])))
                cn0 = _parse_float(raw[SRC_CN0])
                elevation = _parse_float(raw[SRC_ELEVATION])
                azimuth = _parse_float(raw[SRC_AZIMUTH])
                residual = _parse_float(raw[SRC_RESIDUAL])
                rate_consistency = _parse_float(raw[SRC_RATE_CONSISTENCY])
                label_raw = int(round(_parse_float(raw[SRC_LABEL])))
            except (TypeError, ValueError, KeyError):
                stats.bad_value += 1
                continue

            # --- 过滤哨兵值：无法计算变化率一致性的行 ---
            if rate_consistency == rate_sentinel:
                stats.rate_sentinel += 1
                continue

            # --- 过滤伪距残差野值 ---
            if abs(residual) >= residual_limit:
                stats.large_residual += 1
                continue

            # --- 标签映射 -1 -> 0、1 -> 1，其余标签视为异常丢弃 ---
            if label_raw not in LABEL_MAP:
                stats.bad_label += 1
                continue
            label = LABEL_MAP[label_raw]

            stats.kept += 1
            rows.append(
                {
                    "location_id": location_id,
                    # GPS_Time(s) 是整秒；整数能整除时写成 int，输出更干净（read_measurements 仍会转 float）。
                    "epoch": int(epoch) if float(epoch).is_integer() else epoch,
                    # PRN 是裸卫星编号，零填充成两位字符串让字典序与数值序一致（src/data.py 按字符串排序 spatial）。
                    "satellite_id": f"{prn:02d}",
                    "label": label,
                    "cn0": cn0,
                    "elevation": elevation,
                    "azimuth": azimuth,
                    "pseudorange_residual": residual,
                }
            )
    return rows


def write_dataset(path: Path, rows: list[dict[str, object]]) -> None:
    """把合并后的行写成训练脚本可直接读取的 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def discover_files(src_dir: Path) -> list[tuple[str, Path]]:
    """在源目录中按 P2..P8 白名单发现 CSV，返回 (location_id, path) 列表。"""

    found: list[tuple[str, Path]] = []
    for location_id in SOURCE_LOCATIONS:
        candidate = src_dir / f"{location_id}.csv"
        if candidate.exists():
            found.append((location_id, candidate))
    if not found:
        raise FileNotFoundError(
            f"No P2..P8 CSV files found under {src_dir}. Check --src-dir."
        )
    return found


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge LOS-NLOS-main P2..P8 static CSVs into the project's training CSV format."
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=Path("data/LOS-NLOS-main/data for sharing_csv"),
        help="Directory containing P2.csv .. P8.csv.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/los_nlos_static.csv"),
        help="Output CSV path in the project's 8-column format.",
    )
    parser.add_argument(
        "--rate-sentinel",
        type=float,
        default=9999.0,
        help="Drop rows whose Pr_rate_consitency equals this sentinel value.",
    )
    parser.add_argument(
        "--residual-limit",
        type=float,
        default=100.0,
        help="Drop rows whose abs(Pseudorange_residual) is >= this limit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = discover_files(args.src_dir)
    print("found source files:", ", ".join(f"{loc}.csv" for loc, _ in files))

    stats = FilterStats()
    all_rows: list[dict[str, object]] = []
    for location_id, path in files:
        all_rows.extend(
            convert_file(
                path,
                location_id,
                rate_sentinel=args.rate_sentinel,
                residual_limit=args.residual_limit,
                stats=stats,
            )
        )

    if not all_rows:
        raise ValueError("No rows survived filtering; nothing to write.")

    # 按 (location_id, epoch, satellite_id) 排序，输出顺序确定可复现，同一历元的卫星相邻排列。
    all_rows.sort(key=lambda row: (row["location_id"], float(row["epoch"]), row["satellite_id"]))
    write_dataset(args.out, all_rows)

    # --- 打印转换摘要，便于核对过滤是否符合预期 ---
    los = sum(1 for row in all_rows if row["label"] == 1)
    nlos = len(all_rows) - los
    per_loc: dict[str, int] = {}
    for row in all_rows:
        per_loc[row["location_id"]] = per_loc.get(row["location_id"], 0) + 1

    print(f"\nwrote {len(all_rows)} rows to {args.out}")
    print(f"read {stats.read} raw rows")
    print(
        "dropped -> "
        f"bad/NaN: {stats.bad_value}, "
        f"rate_consistency==9999: {stats.rate_sentinel}, "
        f"|residual|>={args.residual_limit:g}: {stats.large_residual}, "
        f"bad_label: {stats.bad_label}"
    )
    print(f"labels -> LOS(1): {los}, NLOS(0): {nlos}")
    print("rows per location:", ", ".join(f"{loc}={per_loc[loc]}" for loc in sorted(per_loc)))
    print(
        f"epochs: {len({row['epoch'] for row in all_rows})}; "
        f"satellites: {len({row['satellite_id'] for row in all_rows})}"
    )


if __name__ == "__main__":
    main()
