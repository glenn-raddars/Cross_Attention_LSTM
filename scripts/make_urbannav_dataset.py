from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


GPS_UTC_LEAP_SECONDS = 18
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
    utc_time: int
    latitude: float
    longitude: float


@dataclass(frozen=True)
class SkyMask:
    latitude: float
    longitude: float
    mask: list[float]


@dataclass(frozen=True)
class NMEAMeasurement:
    utc_time: int
    satellite_id: str
    elevation: float
    azimuth: float
    cn0: float


def dms_to_degrees(degrees: str, minutes: str, seconds: str) -> float:
    sign = -1.0 if degrees.startswith("-") else 1.0
    return sign * (abs(float(degrees)) + float(minutes) / 60.0 + float(seconds) / 3600.0)


def read_groundtruth(path: Path) -> dict[int, GroundTruth]:
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
    return min(
        skymasks,
        key=lambda mask: (mask.latitude - point.latitude) ** 2 + (mask.longitude - point.longitude) ** 2,
    )


def nmea_satellite_id(talker: str, raw_svid: str) -> str | None:
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
        # Some phone logs store QZSS SVIDs as signed values. Keep the common
        # RINEX Jxx numbering when it can be recovered unambiguously.
        if svid < 0:
            svid = abs(svid) - 187
        return f"J{svid:02d}" if svid > 0 else None
    return None


def parse_gsv_line(line: str) -> list[NMEAMeasurement]:
    body, _, timestamp_text = line.strip().rpartition(",")
    if not body.startswith("$") or "GSV" not in body[:6]:
        return []
    try:
        utc_time = int(round(int(timestamp_text) / 1000.0))
    except ValueError:
        return []

    fields = body.split(",")
    talker = fields[0][1:3]
    measurements: list[NMEAMeasurement] = []
    for start in range(4, len(fields) - 3, 4):
        raw_svid, elevation, azimuth, cn0_and_checksum = fields[start : start + 4]
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
                    azimuth=float(azimuth) % 360.0,
                    cn0=float(cn0),
                )
            )
        except ValueError:
            continue
    return measurements


def read_nmea(path: Path) -> dict[tuple[int, str], NMEAMeasurement]:
    by_time_sat: dict[tuple[int, str], NMEAMeasurement] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            for measurement in parse_gsv_line(line):
                by_time_sat[(measurement.utc_time, measurement.satellite_id)] = measurement
    if not by_time_sat:
        raise ValueError(f"No GSV measurements were parsed from {path}")
    return by_time_sat


def normalize_rinex_satellite_id(text: str) -> str:
    system = text[0]
    prn = int(text[1:].strip())
    return f"{system}{prn:02d}"


def rinex_epoch_to_utc(parts: list[str]) -> int:
    year, month, day, hour, minute = [int(value) for value in parts[:5]]
    seconds = float(parts[5])
    whole_seconds = int(seconds)
    dt = datetime(year, month, day, hour, minute, whole_seconds, tzinfo=timezone.utc)
    return int(dt.timestamp()) - GPS_UTC_LEAP_SECONDS


def read_rinex_pseudorange(path: Path) -> dict[tuple[int, str], float]:
    obs_types: dict[str, list[str]] = {}
    pseudoranges: dict[tuple[int, str], float] = {}
    current_time: int | None = None

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "SYS / # / OBS TYPES" in line:
                system = line[0]
                count = int(line[3:6])
                obs_types.setdefault(system, [])
                obs_types[system].extend(line[7:60].split())
                while len(obs_types[system]) < count:
                    continuation = next(handle)
                    obs_types[system].extend(continuation[7:60].split())
                obs_types[system] = obs_types[system][:count]
                continue
            if "END OF HEADER" in line:
                break

        for line in handle:
            if line.startswith(">"):
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
    tracks: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for (utc_time, satellite_id), pseudorange in pseudoranges.items():
        tracks[satellite_id].append((utc_time, pseudorange))

    residuals: dict[tuple[int, str], float] = {}
    for satellite_id, track in tracks.items():
        track.sort()
        rates = [
            (current_range - previous_range) / (current_time - previous_time)
            for (previous_time, previous_range), (current_time, current_range) in zip(track, track[1:])
            if current_time > previous_time
        ]
        typical_rate = sorted(rates)[len(rates) // 2] if rates else 0.0
        previous_time: int | None = None
        previous_range: float | None = None
        for utc_time, pseudorange in track:
            if previous_time is None or previous_range is None or utc_time <= previous_time:
                residual = 0.0
            else:
                dt = utc_time - previous_time
                # One-step code-range residual against the satellite track's
                # typical range rate. This captures abrupt jumps without
                # requiring broadcast ephemeris or an RTK solution.
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
        azimuth_index = int(round(measurement.azimuth)) % 361
        horizon_elevation = mask[azimuth_index]
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
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
    gnss_dir = args.root / "1_UrbanNav-HK-Medium-Urban-1-GNSS"
    stem = f"UrbanNav-HK-Medium-Urban-1.{args.receiver}"
    nmea_path = gnss_dir / f"{stem}.nmea"
    obs_path = gnss_dir / f"{stem}.obs"
    groundtruth_path = args.root / "UrbanNav_TST_GT_raw.txt"
    skymask_path = args.root / "UrbanNav-HK-Medium-Urban-TSTE.csv"

    nmea = read_nmea(nmea_path)
    pseudoranges = read_rinex_pseudorange(obs_path)
    residuals = pseudorange_residuals(pseudoranges)
    groundtruth = read_groundtruth(groundtruth_path)
    skymasks = read_skymasks(skymask_path)
    rows = build_dataset(
        location_id=args.location_id,
        nmea=nmea,
        residuals=residuals,
        groundtruth=groundtruth,
        skymasks=skymasks,
    )
    write_dataset(args.out, rows)

    los_count = sum(1 for row in rows if int(row["label"]) == 1)
    nlos_count = len(rows) - los_count
    print(f"wrote {len(rows)} rows to {args.out}")
    print(f"LOS rows: {los_count}; NLOS rows: {nlos_count}")
    print(f"epochs: {len({row['epoch'] for row in rows})}; satellites: {len({row['satellite_id'] for row in rows})}")


if __name__ == "__main__":
    main()
