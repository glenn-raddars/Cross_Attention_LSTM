from __future__ import annotations

"""把 UrbanNav Medium Urban 原始文件转换成本项目训练脚本需要的 CSV。

本项目的模型入口只认识一张“扁平观测表”：

    location_id, epoch, satellite_id, label, cn0, elevation, azimuth, pseudorange_residual

而 UrbanNav 下载下来的数据是分散在几个文件里的：

1. `.nmea`
   手机/接收机输出的 NMEA 文本。这里主要使用 GSV 语句，因为 GSV 会列出每颗可见卫星的
   仰角 elevation、方位角 azimuth 和 C/N0，即模型输入中的前三个观测特征。

2. `.obs`
   RINEX observation 文件。这里读取 C1C 伪距观测，然后为每颗卫星构造一个可复现的
   “局部伪距残差”特征，作为模型输入中的 pseudorange_residual。

3. `UrbanNav_TST_GT_raw.txt`
   NovAtel SPAN-CPT/IE 后处理 groundtruth。这里使用其中的 UTC 时间和经纬度，
   把每个观测历元映射到车辆真实位置。

4. `UrbanNav-HK-Medium-Urban-TSTE.csv`
   Skymask 文件。每一行是一个位置点，以及该位置下 0..360 度方位角对应的建筑遮挡仰角。
   当卫星仰角高于该方位角的遮挡仰角时，认为是 LOS，否则认为是 NLOS。

最终流程是：

    NMEA GSV 观测 + RINEX C1C 伪距残差 + GroundTruth 位置 + Skymask 遮挡角
    -> 每颗卫星每个历元一行训练 CSV
"""

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


# UrbanNav README 说明 2021 年 5 月数据的 GPS-UTC leap seconds 为 18 秒。
# RINEX epoch 是 GPS time scale；groundtruth 文件第一列是 Unix UTC 秒。
# 因此读取 RINEX 时间后需要减去 18 秒，才能和 groundtruth/NMEA 的 UTC 时间对齐。
GPS_UTC_LEAP_SECONDS = 18

# UrbanNav 官方主 README 提到 Medium Urban 中一个 F9P 接收机在某时段故障。
# GitHub issue #11 进一步指出 ublox.f9p.obs 在下面 GPS 时间段没有 observation：
#   2021-05-17 02:42:30.005 GPS time 到 2021-05-17 02:44:41.005 GPS time
# 本脚本不会自动切换到 splitter/m8t 等其他接收机补齐，只会在该时段缺少 RINEX
# 伪距残差时跳过对应样本。
F9P_BLACKOUT_NOTE = (
    "warning: receiver 'ublox.f9p' has a known UrbanNav Medium blackout in "
    "2021-05-17 02:42:30.005 to 02:44:41.005 GPS time. This script does not "
    "substitute another receiver; rows without matching RINEX pseudorange "
    "residuals will be skipped. Consider --receiver ublox.f9p.splitter or "
    "another receiver if you need a more continuous sequence."
)

# 输出列必须和 src/data.py / src/data_np.py 中 REQUIRED_COLUMNS、FEATURE_COLUMNS 对齐。
# 训练代码会根据这些列名读取、排序、构造 temporal/spatial 两路输入。
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


@dataclass(frozen=True)
class GroundTruth:
    """某一秒车辆的真实位置。

    utc_time:
        Unix UTC 秒。用 int 是因为当前 GNSS/NMEA/GT 都近似 1Hz，对齐时按秒即可。
    latitude/longitude:
        十进制度。原始 GT 文件使用 D M S 形式，读取时会转换。
    """

    utc_time: int
    latitude: float
    longitude: float


@dataclass(frozen=True)
class SkyMask:
    """某个位置点的天空遮挡轮廓。

    mask 是长度 361 的数组：mask[azimuth] 表示该整数方位角上的遮挡仰角。
    例如 mask[90] = 45 表示正东方向上，45 度以下可能被建筑遮挡。
    """

    latitude: float
    longitude: float
    mask: list[float]


@dataclass(frozen=True)
class NMEAMeasurement:
    """从一条 GSV 语句解析出来的单颗卫星观测。"""

    utc_time: int
    satellite_id: str
    elevation: float
    azimuth: float
    cn0: float


def dms_to_degrees(degrees: str, minutes: str, seconds: str) -> float:
    """把 groundtruth 中的 D M S 经纬度转换成十进制度。

    UrbanNav_TST_GT_raw.txt 里经纬度格式类似：

        22 18 04.31949
        114 10 44.60559

    十进制度计算公式为 D + M / 60 + S / 3600。
    """

    sign = -1.0 if degrees.startswith("-") else 1.0
    return sign * (abs(float(degrees)) + float(minutes) / 60.0 + float(seconds) / 3600.0)


def read_groundtruth(path: Path) -> dict[int, GroundTruth]:
    """读取 groundtruth 文件，返回 utc_time -> GroundTruth。

    原始文件前两行是表头和单位说明，无法转成 float，因此这里靠 try/except 跳过。
    有效数据列中：

    - parts[0]: UTC Unix 秒
    - parts[3:6]: 纬度 D M S
    - parts[6:9]: 经度 D M S

    只保留这几个字段，因为生成 LOS/NLOS 标签只需要“某一秒车辆在哪里”。
    """

    groundtruth: dict[int, GroundTruth] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                utc_time = int(round(float(parts[0])))
                latitude = dms_to_degrees(parts[3], parts[4], parts[5])
                longitude = dms_to_degrees(parts[6], parts[7], parts[8])
            except ValueError:
                continue
            groundtruth[utc_time] = GroundTruth(utc_time, latitude, longitude)
    if not groundtruth:
        raise ValueError(f"No groundtruth rows were parsed from {path}")
    return groundtruth


def read_skymasks(path: Path) -> list[SkyMask]:
    """读取 skymask CSV。

    UrbanNav 的 Medium Urban skymask 表头写作：

        Latitude(deg),Longitude(deg),Altitude(m),Skymask(1*361,deg)

    实际 CSV 是 364 列：

    - 第 0 列: 纬度
    - 第 1 列: 经度
    - 第 2 列: 高程，这里不用
    - 第 3..363 列: 0..360 度方位角对应的遮挡仰角

    注意 Python 切片右端不包含，所以 row[3:364] 正好取到 361 个 mask 值。
    """

    skymasks: list[SkyMask] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 364:
                continue
            try:
                skymasks.append(SkyMask(float(row[0]), float(row[1]), [float(value) for value in row[3:364]]))
            except ValueError:
                continue
    if not skymasks:
        raise ValueError(f"No skymask rows were parsed from {path}")
    return skymasks


def nearest_skymask(point: GroundTruth, skymasks: list[SkyMask]) -> SkyMask:
    """为车辆当前位置选择最近的 skymask 行。

    skymask 文件是一张空间采样网格，并不是每个 GT 点都有完全相同的经纬度。
    这里使用经纬度平方距离选最近点。由于同一场景范围较小，只是做最近邻查找，
    不需要把经纬度投影到米制坐标也足够稳定。
    """

    return min(
        skymasks,
        key=lambda mask: (mask.latitude - point.latitude) ** 2 + (mask.longitude - point.longitude) ** 2,
    )


def nmea_satellite_id(talker: str, raw_svid: str) -> str | None:
    """把 NMEA 中的 talker + SVID 转成 RINEX 风格卫星编号。

    NMEA 的 GSV 语句里，卫星系统由 talker 表示：

    - GP: GPS -> Gxx
    - GL: GLONASS -> Rxx
    - GA: Galileo -> Exx
    - GB: BeiDou -> Cxx
    - GQ: QZSS -> Jxx

    这样做是为了和 RINEX `.obs` 文件中的卫星编号保持一致，后面才能用
    (utc_time, satellite_id) 把 NMEA 特征和 RINEX 伪距残差合并起来。
    """

    try:
        svid = int(raw_svid)
    except ValueError:
        return None
    if talker == "GP":
        return f"G{svid:02d}"
    if talker == "GL":
        return f"R{svid - 64:02d}" if svid > 64 else f"R{svid:02d}"
    if talker == "GA":
        return f"E{svid:02d}"
    if talker == "GB":
        return f"C{svid:02d}"
    if talker == "GQ":
        # 部分手机日志会把 QZSS 的 SVID 写成负数，例如 -189。
        # 这里按常见映射恢复到 Jxx 编号，恢复不了时返回 None 跳过。
        if svid < 0:
            svid = abs(svid) - 187
        return f"J{svid:02d}" if svid > 0 else None
    return None


def parse_gsv_line(line: str) -> list[NMEAMeasurement]:
    """解析一行 NMEA，如果是 GSV 则返回其中所有卫星观测。

    UrbanNav 手机 NMEA 文件的每行末尾额外带有一个毫秒级 Unix 时间戳，例如：

        $GPGSV,4,1,13,01,40,168,24,...*68,1621218691980

    标准 GSV 主体中，每颗卫星占 4 个字段：

        SVID, elevation, azimuth, C/N0

    一条 GSV 最多可包含 4 颗卫星，所以本函数会返回 list。
    非 GSV 语句，例如 GGA/RMC/GSA，直接返回空列表。
    """

    # rpartition(",") 从最后一个逗号切开，避免 NMEA 主体内部的逗号干扰。
    # body 是标准 NMEA 语句，timestamp_text 是 UrbanNav 额外附加的毫秒时间戳。
    body, _, timestamp_text = line.strip().rpartition(",")
    if not body.startswith("$") or "GSV" not in body[:6]:
        return []
    try:
        # NMEA 末尾时间戳单位是毫秒；训练数据按秒对齐 groundtruth 和 RINEX。
        utc_time = int(round(int(timestamp_text) / 1000.0))
    except ValueError:
        return []

    fields = body.split(",")
    # "$GPGSV" -> "GP"，"$GAGSV" -> "GA"。
    talker = fields[0][1:3]
    measurements: list[NMEAMeasurement] = []
    for start in range(4, len(fields) - 3, 4):
        raw_svid, elevation, azimuth, cn0_and_checksum = fields[start : start + 4]
        # C/N0 字段可能带有校验和，例如 "24*68"，这里只取星号前面的数值。
        cn0 = cn0_and_checksum.split("*", 1)[0]
        if not raw_svid or not elevation or not azimuth or not cn0:
            continue
        satellite_id = nmea_satellite_id(talker, raw_svid)
        if satellite_id is None:
            continue
        try:
            measurements.append(
                NMEAMeasurement(
                    utc_time=utc_time,
                    satellite_id=satellite_id,
                    elevation=float(elevation),
                    # 方位角归一到 [0, 360)，便于后续作为 skymask 索引。
                    azimuth=float(azimuth) % 360.0,
                    cn0=float(cn0),
                )
            )
        except ValueError:
            continue
    return measurements


def read_nmea(path: Path) -> dict[tuple[int, str], NMEAMeasurement]:
    """读取整个 NMEA 文件，返回 (utc_time, satellite_id) -> NMEAMeasurement。

    同一秒同一颗卫星可能在不同 GSV 分组中重复出现。这里直接用字典覆盖，
    保留最后一次出现的观测。对于 1Hz 训练表来说，这样能保证每秒每星唯一一行。
    """

    by_time_sat: dict[tuple[int, str], NMEAMeasurement] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            for measurement in parse_gsv_line(line):
                by_time_sat[(measurement.utc_time, measurement.satellite_id)] = measurement
    if not by_time_sat:
        raise ValueError(f"No GSV measurements were parsed from {path}")
    return by_time_sat


def normalize_rinex_satellite_id(text: str) -> str:
    """把 RINEX 观测行开头的卫星编号规范化成 G01/E26 这种两位格式。"""

    system = text[0]
    prn = int(text[1:].strip())
    return f"{system}{prn:02d}"


def rinex_epoch_to_utc(parts: list[str]) -> int:
    """把 RINEX epoch 行中的 GPS 时间转换为 Unix UTC 秒。

    RINEX epoch 行类似：

        > 2021  5 17  2 33 13.4446064  0  9

    这里先按 UTC datetime 生成 Unix 秒，再减去 GPS-UTC leap seconds。
    这样得到的时间才能和 groundtruth 第一列、NMEA 末尾时间戳对齐。
    """

    year, month, day, hour, minute = [int(value) for value in parts[:5]]
    seconds = float(parts[5])
    # 当前训练表按整秒对齐，忽略小数秒。UrbanNav 手机 RINEX 和 NMEA 都约为 1Hz。
    whole_seconds = int(seconds)
    dt = datetime(year, month, day, hour, minute, whole_seconds, tzinfo=timezone.utc)
    return int(dt.timestamp()) - GPS_UTC_LEAP_SECONDS


def read_rinex_pseudorange(path: Path) -> dict[tuple[int, str], float]:
    """从 RINEX observation 文件读取 C1C 伪距。

    RINEX 3.x 的一个关键点是：每个卫星系统的观测字段顺序可能不同，需要先读头部的
    `SYS / # / OBS TYPES`。例如 Google Pixel 4 文件里 GPS 有：

        G    8 C1C L1C D1C S1C C5Q L5Q D5Q S5Q

    这说明 GPS 每颗卫星一行中，第 0 个 16 字符字段是 C1C。
    解析体数据时就按这个字段顺序和固定宽度去切片。

    返回值仍然用 (utc_time, satellite_id) 做 key，方便和 NMEA 数据合并。
    """

    obs_types: dict[str, list[str]] = {}
    pseudoranges: dict[tuple[int, str], float] = {}
    current_time: int | None = None

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        # 第一段：读取 RINEX 头部，建立每个卫星系统的观测类型列表。
        for line in handle:
            if "SYS / # / OBS TYPES" in line:
                system = line[0]
                count = int(line[3:6])
                obs_types.setdefault(system, [])
                obs_types[system].extend(line[7:60].split())
                # RINEX 头部一行最多只能放有限个观测类型，类型很多时会续行。
                # 所以这里持续读取，直到收齐 count 个观测类型。
                while len(obs_types[system]) < count:
                    continuation = next(handle)
                    obs_types[system].extend(continuation[7:60].split())
                obs_types[system] = obs_types[system][:count]
                continue
            if "END OF HEADER" in line:
                break

        # 第二段：读取每个 epoch 下的卫星观测。
        for line in handle:
            if line.startswith(">"):
                # epoch 行只更新时间；后续卫星行都属于这个 current_time。
                parts = line[1:].split()
                current_time = rinex_epoch_to_utc(parts)
                continue
            if current_time is None or len(line) < 4:
                continue
            satellite_id = normalize_rinex_satellite_id(line[:3])
            system = satellite_id[0]
            types = obs_types.get(system, [])
            if "C1C" not in types:
                continue
            index = types.index("C1C")
            # RINEX observation value 是固定宽度 16 字符字段。
            # 卫星编号占前 3 字符，所以第 index 个观测值从 3 + index * 16 开始。
            start = 3 + index * 16
            value_text = line[start : start + 14].strip()
            if not value_text:
                continue
            try:
                pseudoranges[(current_time, satellite_id)] = float(value_text)
            except ValueError:
                continue
    if not pseudoranges:
        raise ValueError(f"No C1C pseudoranges were parsed from {path}")
    return pseudoranges


def pseudorange_residuals(pseudoranges: dict[tuple[int, str], float]) -> dict[tuple[int, str], float]:
    """根据每颗卫星的 C1C 伪距序列构造局部伪距残差。

    论文输入需要 pseudorange residual / PRE。严格意义上的 PRE 通常来自 GNSS 解算：
    需要广播星历、接收机钟差、卫星钟差、定位解等。当前下载内容里没有直接提供
    RTKLIB/GraphGNSSLib 后处理残差，因此这里构造一个完全可复现的替代特征：

    1. 按卫星分组，得到该卫星随时间变化的伪距序列。
    2. 计算相邻历元的伪距变化率。
    3. 用中位数变化率作为该卫星在该片段的“典型变化率”。
    4. 对每个历元做一步预测：

           predicted_range = previous_range + typical_rate * dt

       当前伪距和预测伪距的绝对差值就是局部残差。

    这个残差能反映码伪距突跳、遮挡/多路径造成的异常变化。它不是严格 RTK 残差，
    但可以稳定提供模型所需的第四个输入特征，并且不依赖额外星历下载。
    """

    tracks: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for (utc_time, satellite_id), pseudorange in pseudoranges.items():
        tracks[satellite_id].append((utc_time, pseudorange))

    residuals: dict[tuple[int, str], float] = {}
    for satellite_id, track in tracks.items():
        # 每颗卫星独立处理，因为不同卫星的几何距离和运动趋势不同。
        track.sort()
        rates = [
            (current_range - previous_range) / (current_time - previous_time)
            for (previous_time, previous_range), (current_time, current_range) in zip(track, track[1:])
            if current_time > previous_time
        ]
        # 用中位数而不是均值，降低伪距突跳对“典型变化率”的影响。
        typical_rate = sorted(rates)[len(rates) // 2] if rates else 0.0
        previous_time: int | None = None
        previous_range: float | None = None
        for utc_time, pseudorange in track:
            if previous_time is None or previous_range is None or utc_time <= previous_time:
                # 每条卫星轨迹的第一个点没有历史可预测，残差置 0。
                residual = 0.0
            else:
                dt = utc_time - previous_time
                predicted = previous_range + typical_rate * dt
                residual = abs(pseudorange - predicted)
            residuals[(utc_time, satellite_id)] = residual
            previous_time = utc_time
            previous_range = pseudorange
    return residuals


def build_dataset(
    *,
    location_id: str,
    nmea: dict[tuple[int, str], NMEAMeasurement],
    residuals: dict[tuple[int, str], float],
    groundtruth: dict[int, GroundTruth],
    skymasks: list[SkyMask],
) -> list[dict[str, str | float | int]]:
    """合并所有来源，生成训练 CSV 的行列表。

    合并条件是 `(utc_time, satellite_id)`：

    - NMEA 提供 cn0/elevation/azimuth
    - RINEX 残差提供 pseudorange_residual
    - groundtruth 提供这个 utc_time 的车辆位置
    - skymask 根据车辆位置提供遮挡仰角，用来生成 label

    如果某个 NMEA 观测找不到 groundtruth 或 RINEX 残差，就跳过。这样输出表中
    每一行都是完整训练样本，不需要训练代码再处理缺失值。
    """

    # 同一秒所有卫星的车辆位置相同，因此最近 skymask 也相同。
    # 缓存可以避免对同一个 utc_time 重复做最近邻搜索。
    mask_cache: dict[int, SkyMask] = {}
    rows: list[dict[str, str | float | int]] = []
    for key, measurement in sorted(nmea.items()):
        utc_time, satellite_id = key
        if utc_time not in groundtruth:
            continue
        if key not in residuals:
            continue
        if utc_time not in mask_cache:
            mask_cache[utc_time] = nearest_skymask(groundtruth[utc_time], skymasks)
        mask = mask_cache[utc_time].mask

        # skymask 的索引是整数方位角 0..360；NMEA 方位角也是角度，这里四舍五入。
        # Python 的 % 361 可以兜住 360 度这个合法索引。
        azimuth_index = int(round(measurement.azimuth)) % 361
        horizon_elevation = mask[azimuth_index]

        # label 约定来自 README_CN.md 和训练指标：
        #   1 = LOS
        #   0 = NLOS
        # 如果卫星仰角高于建筑遮挡仰角，就认为卫星在可见天空区域内。
        label = 1 if measurement.elevation > horizon_elevation else 0
        rows.append(
            {
                "location_id": location_id,
                "epoch": utc_time,
                "satellite_id": satellite_id,
                "label": label,
                "cn0": measurement.cn0,
                "elevation": measurement.elevation,
                "azimuth": measurement.azimuth,
                "pseudorange_residual": residuals[key],
            }
        )
    if not rows:
        raise ValueError("No rows were generated. Check that NMEA, RINEX, and groundtruth times overlap.")
    return rows


def write_dataset(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    """把生成的行写成训练脚本可直接读取的 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """命令行参数。

    默认值匹配当前项目中的下载目录：

        data/urbannav-medium/
        data/urbannav-medium/1_UrbanNav-HK-Medium-Urban-1-GNSS/

    `--receiver` 用于选择同一场景下不同接收机/手机的数据，例如：

        google.pixel4
        huawei.p40pro
        samsung.note8
        xiaomi.mi8
        ublox.f9p

    前提是对应的 `.nmea` 和 `.obs` 都存在。
    """

    parser = argparse.ArgumentParser(description="Convert UrbanNav Medium Urban files to the training CSV format.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/urbannav-medium"),
        help="Directory containing the downloaded UrbanNav medium dataset.",
    )
    parser.add_argument("--receiver", default="google.pixel4", help="Receiver name embedded in the .nmea/.obs filenames.")
    parser.add_argument("--location-id", default="UrbanNav-HK-Medium-Urban-1")
    parser.add_argument("--out", type=Path, default=Path("data/urbannav_medium_google_pixel4.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.receiver == "ublox.f9p":
        print(F9P_BLACKOUT_NOTE)

    # 根据 UrbanNav Medium Urban 的实际目录和文件命名规则拼出四个输入文件路径。
    gnss_dir = args.root / "1_UrbanNav-HK-Medium-Urban-1-GNSS"
    stem = f"UrbanNav-HK-Medium-Urban-1.{args.receiver}"
    nmea_path = gnss_dir / f"{stem}.nmea"
    obs_path = gnss_dir / f"{stem}.obs"
    groundtruth_path = args.root / "UrbanNav_TST_GT_raw.txt"
    skymask_path = args.root / "UrbanNav-HK-Medium-Urban-TSTE.csv"

    # 逐个解析原始文件。这里保持中间结果为 dict/list，便于理解和调试。
    nmea = read_nmea(nmea_path)
    pseudoranges = read_rinex_pseudorange(obs_path)
    residuals = pseudorange_residuals(pseudoranges)
    groundtruth = read_groundtruth(groundtruth_path)
    skymasks = read_skymasks(skymask_path)

    # 最后一步才真正合并不同来源，并生成训练代码所需的 8 列 CSV。
    rows = build_dataset(
        location_id=args.location_id,
        nmea=nmea,
        residuals=residuals,
        groundtruth=groundtruth,
        skymasks=skymasks,
    )
    write_dataset(args.out, rows)

    # 打印一个简短摘要，帮助确认转换是否正常：行数、类别分布、历元数、卫星数。
    los_count = sum(1 for row in rows if int(row["label"]) == 1)
    nlos_count = len(rows) - los_count
    print(f"wrote {len(rows)} rows to {args.out}")
    print(f"LOS rows: {los_count}; NLOS rows: {nlos_count}")
    print(f"epochs: {len({row['epoch'] for row in rows})}; satellites: {len({row['satellite_id'] for row in rows})}")


if __name__ == "__main__":
    main()
