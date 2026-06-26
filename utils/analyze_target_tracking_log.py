#!/usr/bin/env python3
"""Plot target-tracking telemetry emitted by moving_target_publisher."""

import argparse
import json
import re
from pathlib import Path

import numpy as np


FLOAT_RE = re.compile(r"(?<![A-Za-z])[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
TIMESTAMP_RE = re.compile(r"\[(\d+(?:\.\d+)?)\]")
STATE_PREFIX = "Updated state ="
TARGET_OBSERVATION_PREFIX = "Received target observation: position="
TARGET_UPDATE_PREFIX = "Target update:"
COLLISION_PREFIX = "Reached moving target:"
TELEMETRY_PREFIX = "TRACKING_TELEMETRY "
TELEMETRY_COLLISION_PREFIX = "TRACKING_COLLISION "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot vehicle/target trajectories, collisions, and attitude tracking. "
            "New moving_target_publisher logs are self-contained."
        )
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="MPC log for legacy parsing, or a target telemetry log when used alone.",
    )
    parser.add_argument(
        "--target-log",
        type=Path,
        default=None,
        help="moving_target_publisher log; preferred source for new telemetry.",
    )
    parser.add_argument(
        "--save-plot",
        type=Path,
        default=None,
        help="Output image path for the 3D trajectory plot.",
    )
    parser.add_argument(
        "--save-attitude-plot",
        type=Path,
        default=None,
        help="Output image path for the desired-versus-observed attitude plot.",
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Do not open interactive windows."
    )
    args = parser.parse_args()
    if args.log is None and args.target_log is None:
        parser.error("provide --target-log, --log, or both")
    return args


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def extract_floats(text: str) -> list[float]:
    normalized = re.sub(r"np\.float\d+\(", "(", text)
    return [float(value) for value in FLOAT_RE.findall(normalized)]


def extract_timestamp(line: str) -> float | None:
    matches = TIMESTAMP_RE.findall(line)
    return float(matches[-1]) if matches else None


def extract_prefixed_chunks(lines: list[str], prefix: str):
    index = 0
    while index < len(lines):
        line = lines[index]
        if prefix not in line:
            index += 1
            continue
        timestamp = extract_timestamp(line)
        chunk = line.split(prefix, maxsplit=1)[1]
        while chunk.count("[") > chunk.count("]") and index + 1 < len(lines):
            index += 1
            chunk += " " + lines[index].strip()
        yield timestamp, chunk
        index += 1


def parse_json_records(path: Path, prefix: str) -> list[dict]:
    records = []
    for line in read_lines(path):
        if prefix not in line:
            continue
        try:
            payload = json.loads(line.split(prefix, maxsplit=1)[1])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def records_to_array(records: list[dict], key: str, columns: int) -> np.ndarray:
    values = []
    for record in records:
        value = np.asarray(record.get(key, []), dtype=float)
        if value.shape == (columns,) and np.all(np.isfinite(value)):
            values.append(value)
    return np.asarray(values, dtype=float).reshape(-1, columns)


def parse_telemetry(path: Path) -> dict[str, np.ndarray] | None:
    records = parse_json_records(path, TELEMETRY_PREFIX)
    if not records:
        return None
    time_sec = np.asarray([record.get("time_sec", np.nan) for record in records], dtype=float)
    vehicle = records_to_array(records, "vehicle_position_ned_m", 3)
    target = records_to_array(records, "target_position_ned_m", 3)
    desired_euler = records_to_array(records, "desired_euler_deg", 3)
    observed_euler = records_to_array(records, "observed_euler_deg", 3)
    sample_count = min(
        len(time_sec), len(vehicle), len(target), len(desired_euler), len(observed_euler)
    )
    if sample_count == 0:
        return None
    return {
        "time_sec": time_sec[:sample_count],
        "vehicle": vehicle[:sample_count],
        "target": target[:sample_count],
        "desired_euler_deg": desired_euler[:sample_count],
        "observed_euler_deg": observed_euler[:sample_count],
    }


def parse_telemetry_collisions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    records = parse_json_records(path, TELEMETRY_COLLISION_PREFIX)
    vehicle = records_to_array(records, "vehicle_position_ned_m", 3)
    target = records_to_array(records, "target_position_ned_m", 3)
    timestamps = np.asarray([record.get("time_sec", np.nan) for record in records], dtype=float)
    count = min(len(vehicle), len(target), len(timestamps))
    return vehicle[:count], target[:count], timestamps[:count]


def parse_legacy_vehicle(path: Path) -> np.ndarray:
    positions = []
    for _, chunk in extract_prefixed_chunks(read_lines(path), STATE_PREFIX):
        values = extract_floats(chunk)
        if len(values) >= 3:
            positions.append(values[:3])
    if not positions:
        raise ValueError(f'No vehicle states found in "{path}".')
    return np.asarray(positions, dtype=float)


def position_values(text: str) -> list[float]:
    position_text = text.split("position=", maxsplit=1)[-1]
    for delimiter in (", velocity=", ", radius=", ", target=", ", distance="):
        position_text = position_text.split(delimiter, maxsplit=1)[0]
    return extract_floats(position_text)


def parse_legacy_target(path: Path) -> np.ndarray:
    lines = read_lines(path)
    positions = []
    for _, chunk in extract_prefixed_chunks(lines, TARGET_OBSERVATION_PREFIX):
        values = extract_floats(chunk.split(", radius=", maxsplit=1)[0])
        if len(values) >= 3:
            positions.append(values[:3])
    if not positions:
        for _, chunk in extract_prefixed_chunks(lines, TARGET_UPDATE_PREFIX):
            values = position_values(chunk)
            if len(values) >= 3:
                positions.append(values[:3])
    if not positions:
        raise ValueError(f'No target positions found in "{path}".')
    return np.asarray(positions, dtype=float)


def parse_legacy_collisions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vehicle_positions, target_positions, timestamps = [], [], []
    for timestamp, chunk in extract_prefixed_chunks(read_lines(path), COLLISION_PREFIX):
        position_part, separator, remainder = chunk.partition(", target=")
        if not separator:
            continue
        vehicle_values = position_values(position_part)
        target_values = extract_floats(remainder.split(", distance=", maxsplit=1)[0])
        if len(vehicle_values) >= 3 and len(target_values) >= 3:
            vehicle_positions.append(vehicle_values[:3])
            target_positions.append(target_values[:3])
            timestamps.append(np.nan if timestamp is None else timestamp)
    return (
        np.asarray(vehicle_positions, dtype=float).reshape(-1, 3),
        np.asarray(target_positions, dtype=float).reshape(-1, 3),
        np.asarray(timestamps, dtype=float),
    )


def ned_to_plot(points: np.ndarray) -> np.ndarray:
    return np.column_stack((points[:, 1], points[:, 0], -points[:, 2]))


def set_axes_equal(ax, points: np.ndarray) -> None:
    mins, maxs = np.min(points, axis=0), np.max(points, axis=0)
    centers = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)), 1.0) / 2.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def plot_trajectories(
    vehicle: np.ndarray,
    target: np.ndarray,
    collision_vehicle: np.ndarray,
    save_path: Path | None,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    vehicle_plot, target_plot = ned_to_plot(vehicle), ned_to_plot(target)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*vehicle_plot.T, linewidth=2.0, label="Quadcopter")
    ax.plot(*target_plot.T, "--", linewidth=2.0, label="Target")
    groups = [vehicle_plot, target_plot]
    if len(collision_vehicle):
        collision_plot = ned_to_plot(collision_vehicle)
        groups.append(collision_plot)
        ax.scatter(*collision_plot[0], marker="*", s=220, color="red", edgecolors="black", label="First collision")
        if len(collision_plot) > 1:
            ax.scatter(*collision_plot[1:].T, marker="x", s=80, color="darkorange", label="Later collisions")
    set_axes_equal(ax, np.vstack(groups))
    ax.set(xlabel="East [m]", ylabel="North [m]", zlabel="Up [m]")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved trajectory plot: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_attitudes(
    time_sec: np.ndarray,
    desired_deg: np.ndarray,
    observed_deg: np.ndarray,
    save_path: Path | None,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    labels = ("Roll", "Pitch", "Yaw")
    desired = desired_deg.copy()
    observed = observed_deg.copy()
    desired[:, 2] = np.rad2deg(np.unwrap(np.deg2rad(desired[:, 2])))
    observed[:, 2] = np.rad2deg(np.unwrap(np.deg2rad(observed[:, 2])))
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for index, axis in enumerate(axes):
        axis.plot(time_sec, desired[:, index], label="Desired", linewidth=1.8)
        axis.plot(time_sec, observed[:, index], label="Observed", linewidth=1.4)
        axis.set_ylabel(f"{labels[index]} [deg]")
        axis.grid(True)
        axis.legend(loc="best")
    axes[-1].set_xlabel("Target publisher time [s]")
    fig.suptitle("Attitude tracking")
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved attitude plot: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def default_attitude_path(trajectory_path: Path | None) -> Path | None:
    if trajectory_path is None:
        return None
    return trajectory_path.with_name(
        f"{trajectory_path.stem}_attitude{trajectory_path.suffix}"
    )


def main() -> None:
    args = parse_args()
    telemetry_log = args.target_log or args.log
    telemetry = parse_telemetry(telemetry_log)

    if telemetry is not None:
        vehicle, target = telemetry["vehicle"], telemetry["target"]
        collision_vehicle, collision_target, collision_times = parse_telemetry_collisions(
            telemetry_log
        )
    else:
        if args.log is None:
            raise ValueError("The supplied target log has no TRACKING_TELEMETRY entries.")
        vehicle = parse_legacy_vehicle(args.log)
        target = parse_legacy_target(args.target_log or args.log)
        collision_vehicle, collision_target, collision_times = parse_legacy_collisions(
            args.log
        )

    print(f"Vehicle points: {len(vehicle)}")
    print(f"Target points: {len(target)}")
    print(f"Collisions: {len(collision_vehicle)}")
    if len(collision_vehicle):
        print(f"First collision vehicle position (NED): {collision_vehicle[0]}")
        print(f"First collision target position (NED): {collision_target[0]}")
        if np.isfinite(collision_times[0]):
            print(f"First collision time: {collision_times[0]:.3f} s")

    plot_trajectories(vehicle, target, collision_vehicle, args.save_plot, not args.no_show)
    if telemetry is not None:
        plot_attitudes(
            telemetry["time_sec"],
            telemetry["desired_euler_deg"],
            telemetry["observed_euler_deg"],
            args.save_attitude_plot or default_attitude_path(args.save_plot),
            not args.no_show,
        )


if __name__ == "__main__":
    main()
