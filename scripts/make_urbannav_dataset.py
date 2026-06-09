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


@dataclass(frozen=True) # 类的初始化常用方法，frozen=True 让实例不可变，hashable，可以作为 dict 的 key。
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
    elevation: float # 仰角
    azimuth: float # 方位角
    cn0: float # 载噪比


def dms_to_degrees(degrees: str, minutes: str, seconds: str) -> float:
    """把 groundtruth 中的 D M S 经纬度转换成十进制度。

    UrbanNav_TST_GT_raw.txt 里经纬度格式类似：

        22 18 04.31949
        114 10 44.60559

    十进制度计算公式为 D + M / 60 + S / 3600。 度分秒中的度数部分可能带有负号，表示南纬或西经。转换时保留这个符号即可。
    """

    sign = -1.0 if degrees.startswith("-") else 1.0 # 负号只看度数部分，分钟和秒数不带符号。
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
            if len(parts) < 9: # 每行至少要有 9 列才能包含上述字段，否则跳过。
                continue
            try:
                utc_time = int(round(float(parts[0]))) # 四舍五入到秒，保持和 NMEA/RINEX 对齐。
                latitude = dms_to_degrees(parts[3], parts[4], parts[5])
                longitude = dms_to_degrees(parts[6], parts[7], parts[8])
            except ValueError: # 跳过第一行表头和第二行单位说明，以及任何格式不正确的行。
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
        skymasks, # 在所有的 skymask 行中，找到距离 GT 点最近的那个。
        key=lambda mask: (mask.latitude - point.latitude) ** 2 + (mask.longitude - point.longitude) ** 2, # lambda函数计算 GT 点和每个 skymask 行之间的经纬度平方距离，min 函数返回距离最近的那个 skymask 行。
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
    body, _, timestamp_text = line.strip().rpartition(",") # line.strip().rpartition(",")返回一个三元组 (body, sep, timestamp_text)，其中 body 是 line 去掉末尾时间戳后的部分，sep 是逗号，timestamp_text 是逗号后面的时间戳文本。
    if not body.startswith("$") or "GSV" not in body[:6]:
        return []
    try:
        # NMEA 末尾时间戳单位是毫秒；训练数据按秒对齐 groundtruth 和 RINEX。
        utc_time = int(round(int(timestamp_text) / 1000.0))
    except ValueError:
        return []

    fields = body.split(",")
    # "$GPGSV" -> "GP"，"$GAGSV" -> "GA"。
    talker = fields[0][1:3] # fields[0] 是 "$GPGSV"，fields[0][1:3] 就是 "GP"，表示卫星系统。
    measurements: list[NMEAMeasurement] = []
    for start in range(4, len(fields) - 3, 4): # 从第 4 个字段开始，每 4 个字段是一颗卫星的观测，直到倒数第 3 个字段为止（因为每颗卫星需要 4 个字段）。合法条件是start + 3 <= len(fields) - 1，即 start <= len(fields) - 4。
        raw_svid, elevation, azimuth, cn0_and_checksum = fields[start : start + 4]
        # C/N0 字段可能带有校验和，例如 "24*68"，这里只取星号前面的数值。
        cn0 = cn0_and_checksum.split("*", 1)[0] # 1 表示只分割一次，得到 C/N0 和校验和两部分，取前面 C/N0 的数值。
        if not raw_svid or not elevation or not azimuth or not cn0:
            continue
        satellite_id = nmea_satellite_id(talker, raw_svid) # 把 talker 和 raw_svid 转成 RINEX 风格的卫星编号，例如 "GP" + "01" -> "G01"。如果无法解析，返回 None。
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
    return f"{system}{prn:02d}" # 例如 "G1" -> "G01"，"E26" -> "E26"，"R64" -> "R64"，"R65" -> "R01"（GLONASS 卫星编号从 1 开始，但 RINEX 可能写成 65 表示 GLONASS-1 号卫星，减去 64 恢复到 R01）。如果文本格式不合法，可能会抛出 ValueError，这里不捕获，让调用者处理。


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
    return int(dt.timestamp()) - GPS_UTC_LEAP_SECONDS # 减去18秒，使 GPS 时间转换为 UTC 时间


def read_rinex_pseudorange(path: Path) -> dict[tuple[int, str], float]:
    """从 RINEX observation 文件读取 C1C 伪距（GPS L1 C/A 码伪距，单位米）。

    作用
    ----
    扫描一个 RINEX 3.x observation 文件，提取每个时刻、每颗卫星的 C1C 伪距，
    返回一个以 ``(utc_time, satellite_id)`` 为 key 的字典，供后续与 NMEA、
    groundtruth 数据按"每秒每星"合并成训练表。

    为什么需要先解析头部
    --------------------
    RINEX 3.x 不像旧版那样固定字段顺序：每个卫星系统（G=GPS、E=Galileo、
    R=GLONASS……）各自声明自己有哪些观测类型、以及它们的排列顺序，写在头部的
    ``SYS / # / OBS TYPES`` 段里。例如 Google Pixel 4 文件中 GPS 这一行是：

        G    8 C1C L1C D1C S1C C5Q L5Q D5Q S5Q

    含义是：GPS 每颗卫星的观测行里依次排着 8 个观测值，第 0 个是 C1C、第 1 个是
    L1C……所以必须先读头部，才能知道 C1C 在每行的第几个字段，进而算出它在固定宽度
    布局中的字符偏移。不同系统（如 Galileo）的字段顺序可能完全不同，因此 obs_types
    要按系统分别保存。

    RINEX observation 体数据的列布局
    --------------------------------
    每条卫星观测行是定宽格式：开头 3 个字符是卫星编号（如 "G01"），其后每个观测值
    占 16 个字符（14 位数值 + 2 位 LLI/信号强度标志）。因此第 index 个观测值的数值
    部分从字符位置 ``3 + index * 16`` 开始，取 14 个字符即可。

    返回
    ----
    ``dict[(utc_time, satellite_id) -> pseudorange_in_meters]``。
    若整份文件一个 C1C 都没解析到，抛 ValueError（多半意味着文件格式不符合预期）。
    """

    # obs_types: 每个卫星系统 -> 其观测类型按文件中声明的顺序排成的列表。
    #   例如 {"G": ["C1C", "L1C", ...], "E": [...]}，用于后面定位 C1C 的列下标。
    obs_types: dict[str, list[str]] = {}
    # pseudoranges: 最终结果，(时间, 卫星) -> C1C 伪距（米）。
    pseudoranges: dict[tuple[int, str], float] = {}
    # current_time: 当前正在处理的 epoch（时刻）。epoch 行只出现一次，
    #   其后连续若干卫星行都共享这个时间，直到遇到下一个 epoch 行。
    current_time: int | None = None

    # errors="ignore"：RINEX 文件偶有非 ASCII/损坏字节，跳过而不让整个解析崩溃。
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        # === 第一段：解析头部，建立每个卫星系统的观测类型顺序表 ===
        for line in handle:
            if "SYS / # / OBS TYPES" in line:
                system = line[0]            # 第 0 列是系统标识字符，如 'G' / 'E' / 'R'
                count = int(line[3:6])      # 第 3~5 列是该系统的观测类型总个数
                obs_types.setdefault(system, [])
                # 观测类型名从第 7 列开始、第 60 列结束（再往后是 "SYS / # / OBS TYPES"
                # 这个标签本身），用 split() 按空白切成 ["C1C", "L1C", ...]。
                obs_types[system].extend(line[7:60].split())
                # RINEX 头部一行最多只能放有限个观测类型（约 13 个），类型很多时会续行，
                # 续行的标签列为空、数据列同样从第 7 列起。这里持续读取下一行补齐，
                # 直到收集满 count 个为止。
                while len(obs_types[system]) < count:
                    continuation = next(handle)
                    obs_types[system].extend(continuation[7:60].split())
                # 截断到正好 count 个，防止续行多 split 出多余 token。
                obs_types[system] = obs_types[system][:count]
                continue
            if "END OF HEADER" in line:
                break  # 头部结束，剩下的 handle 内容都是观测体数据

        # === 第二段：逐行读取观测体数据，提取每个 epoch 下各卫星的 C1C ===
        for line in handle:
            if line.startswith(">"):
                # 以 '>' 开头的是 epoch（时刻）行，本身不含观测值，只用于切换时间。
                # 去掉开头的 '>' 再 split，交给 rinex_epoch_to_utc 转成 Unix UTC 秒。
                parts = line[1:].split()
                current_time = rinex_epoch_to_utc(parts)
                continue
            # 还没遇到任何 epoch（current_time 为空），或行太短不可能含卫星编号，跳过。
            if current_time is None or len(line) < 4:
                continue
            # 前 3 个字符是卫星编号，规范化成 "G01"/"E26" 这种统一两位格式。
            satellite_id = normalize_rinex_satellite_id(line[:3])
            system = satellite_id[0]                 # 取系统字符以查对应的观测顺序
            types = obs_types.get(system, [])
            # 该系统若根本没声明 C1C（或头部里没有这个系统），这颗卫星无伪距可取，跳过。
            if "C1C" not in types:
                continue
            index = types.index("C1C")               # C1C 在该系统观测序列中的列下标
            # 按定宽布局定位：卫星编号占前 3 字符，每个观测值占 16 字符，
            # 因此第 index 个观测值的数值部分从 3 + index * 16 开始，取 14 个字符。
            start = 3 + index * 16
            value_text = line[start : start + 14].strip()
            if not value_text:
                continue  # 该历元这颗星缺测（字段为空），跳过
            try:
                pseudoranges[(current_time, satellite_id)] = float(value_text)
            except ValueError:
                # 数值字段含意外字符无法转 float，丢弃这一条而不中断整体解析。
                continue
    if not pseudoranges:
        # 一条都没解析到通常说明文件不是预期的 RINEX 3.x observation 格式。
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

    # === 第一步：按卫星把 (时间, 伪距) 重新分组成"轨迹" ===
    # 输入 pseudoranges 的 key 是 (utc_time, satellite_id)，是按"时刻×卫星"散开的；
    # 而残差是沿时间方向的差分，必须先把同一颗卫星的所有历元聚到一起。
    # tracks[satellite_id] = [(t0, r0), (t1, r1), ...]，每颗星一条时间序列。
    tracks: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for (utc_time, satellite_id), pseudorange in pseudoranges.items():
        tracks[satellite_id].append((utc_time, pseudorange))

    residuals: dict[tuple[int, str], float] = {}
    for satellite_id, track in tracks.items():
        # 每颗卫星独立处理，因为不同卫星的几何距离和运动趋势不同：
        # 接近天顶的星伪距变化慢、贴近地平的星变化快，不能混在一起估变化率。
        # 字典插入顺序不保证按时间，差分前必须排序；元组排序默认按第 0 项(时间)升序。
        track.sort() # 按时间升序排序，确保后续差分计算正确。每条轨迹的第一个点没有历史可预测，残差置 0。

        # === 第二步：估计该卫星这一段的"典型变化率"(米/秒) ===
        # 对相邻历元两两求斜率 (Δ伪距 / Δt)，得到一组逐点变化率。
        # 条件 current_time > previous_time 既防止除零，也跳过重复时间戳。
        rates = [
            (current_range - previous_range) / (current_time - previous_time)
            for (previous_time, previous_range), (current_time, current_range) in zip(track, track[1:])
            if current_time > previous_time
        ]
        # 用中位数而不是均值，降低伪距突跳/野值对"典型变化率"的影响：
        # 均值会被个别大跳变拉偏，中位数对这种离群点稳健。
        # 注意这是简化的中位数(偶数个时直接取上中位，不做两数平均)，对本特征已足够。
        # rates 为空(该卫星只有 1 个历元)时退化为 0.0。
        typical_rate = sorted(rates)[len(rates) // 2] if rates else 0.0

        # === 第三步：沿时间做一步预测，残差 = |实测 - 预测| ===
        # 思想：如果伪距是"平稳"演化的，用上一历元 + 典型变化率外推就应接近本历元；
        # 一旦发生码伪距突跳、遮挡或多路径，实测就会明显偏离这条平稳外推线，
        # 偏差(残差)随之变大——这正是我们想喂给模型的异常信号。
        previous_time: int | None = None
        previous_range: float | None = None
        for utc_time, pseudorange in track:
            if previous_time is None or previous_range is None or utc_time <= previous_time:
                # 每条卫星轨迹的第一个点没有历史可预测，残差置 0；
                # utc_time <= previous_time 兜底处理重复/乱序时间戳。
                residual = 0.0
            else:
                dt = utc_time - previous_time
                # 用上一历元伪距 + 典型变化率 × 时间间隔，外推出本历元的"预期伪距"。
                predicted = previous_range + typical_rate * dt
                # 实测与预期之差取绝对值：只关心偏离幅度，不分方向。
                residual = abs(pseudorange - predicted)
            residuals[(utc_time, satellite_id)] = residual # 伪距残差特征，供后续与 NMEA 特征、GT 标签合并成训练表。
            # 滚动更新"上一历元"，供下一个点做一步预测(逐点递推)。
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

    本函数是整个数据集构建流程的"汇聚点"：前面的步骤分别从不同原始文件解析出
    NMEA 观测、RINEX 伪距残差、车辆真值轨迹、天空遮挡掩膜；这里把它们按统一的键
    拼成一张扁平训练表，每一行就是一个 (历元, 卫星) 样本，可直接喂给训练代码。

    合并的主键是 `(utc_time, satellite_id)`，四类来源各贡献一部分字段：

    - NMEA       提供 cn0 / elevation / azimuth（卫星观测量，逐星逐历元）
    - RINEX 残差 提供 pseudorange_residual（上一步算好的伪距异常特征）
    - groundtruth 按 utc_time 提供车辆这一秒的真实位置（与卫星无关，整秒共用）
    - skymask    根据车辆位置给出各方位角的建筑遮挡仰角，用来判定 LOS/NLOS 标签

    主循环以 NMEA 观测为驱动：逐条 NMEA 记录去补齐 groundtruth 和残差，任一来源
    缺失就整行跳过（continue）。因此输出表里每一行都是字段完整的训练样本，训练
    代码无需再处理缺失值（NaN）。

    Returns:
        行字典组成的列表。若没有任何观测能完整对齐（如各来源时间戳不重叠），
        抛 ValueError，避免悄悄写出空 CSV。
    """

    # mask_cache: utc_time -> 最近的 SkyMask。
    # 同一秒内所有卫星共享同一个车辆位置，对应的最近邻掩膜自然也相同；用 utc_time
    # 做缓存键，可避免对同一秒里的每颗卫星都重复跑一次 nearest_skymask 最近邻搜索。
    mask_cache: dict[int, SkyMask] = {}
    rows: list[dict[str, str | float | int]] = []

    # 按 (utc_time, satellite_id) 升序遍历：保证输出行顺序确定、可复现，且同一历元的
    # 多颗卫星相邻排列，便于人工核对，也方便下游按时间切窗口。
    for key, measurement in sorted(nmea.items()): # 先按照 utc_time 升序排序，如果 utc_time 相同则按照 satellite_id 升序排序，保证输出行顺序确定、可复现。
        utc_time, satellite_id = key

        # === 对齐检查：三来源都要覆盖到这条 NMEA 观测，否则该行不完整，整行丢弃 ===
        # 1) 缺这一秒的车辆真值 -> 无法定位、取不到 skymask、生成不了标签。
        if utc_time not in groundtruth:
            continue
        # 2) 缺这颗星这一历元的伪距残差 -> 少一个特征列，丢弃以免引入缺失值。
        if key not in residuals:
            continue

        # === 取这一秒车辆位置对应的天空遮挡掩膜（带缓存）===
        if utc_time not in mask_cache:
            mask_cache[utc_time] = nearest_skymask(groundtruth[utc_time], skymasks)
        mask = mask_cache[utc_time].mask  # mask[方位角] = 该方向建筑物的遮挡仰角(度)

        # === 按卫星方位角查该方向的建筑遮挡仰角 ===
        # skymask 索引是整数方位角 0..360；NMEA 方位角是浮点角度，先四舍五入成整数。
        # 再对 361 取模：把可能出现的 360 度安全映射到合法索引（0..360 共 361 个）。
        azimuth_index = int(round(measurement.azimuth)) % 361
        horizon_elevation = mask[azimuth_index]

        # === 生成 LOS/NLOS 标签 ===
        # label 约定来自 README_CN.md 和训练指标：
        #   1 = LOS （视距，卫星直接可见）
        #   0 = NLOS（非视距，信号被建筑遮挡/反射）
        # 判据：卫星实测仰角 > 该方位的建筑遮挡仰角，说明卫星高过天际线、直接可见。
        label = 1 if measurement.elevation > horizon_elevation else 0

        # === 拼出一行训练样本：标识列 + 标签 + 四个特征列 ===
        rows.append(
            {
                "location_id": location_id,             # 采集地点标识，跨域实验区分数据来源
                "epoch": utc_time,                      # 历元(秒)，时间标识
                "satellite_id": satellite_id,           # 卫星编号，样本标识
                "label": label,                         # 训练目标：LOS=1 / NLOS=0
                "cn0": measurement.cn0,                 # 载噪比，特征
                "elevation": measurement.elevation,     # 卫星仰角，特征
                "azimuth": measurement.azimuth,         # 卫星方位角，特征
                "pseudorange_residual": residuals[key], # 伪距残差，特征
            }
        )

    # 一行都没生成，通常意味着各来源时间戳完全不重叠（解析、时区或时间对齐出了问题）。
    # 早抛错比悄悄返回空表更利于定位问题。
    if not rows:
        raise ValueError("No rows were generated. Check that NMEA, RINEX, and groundtruth times overlap.")
    return rows


def write_dataset(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    """把生成的行写成训练脚本可直接读取的 CSV。"""

    # 确保输出目录存在：parents=True 递归创建多级目录，exist_ok=True 让目录已存在时不报错。
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" 交给 csv 模块自己处理换行，避免在 Windows 上写出多余的空行。
    with path.open("w", encoding="utf-8", newline="") as handle:
        # DictWriter 按 OUTPUT_COLUMNS 的顺序把每行字典映射到固定的列，保证列顺序稳定、与训练脚本约定一致。
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()   # 先写表头行（列名）。
        writer.writerows(rows)  # 再一次性写入所有数据行。


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
