#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import numpy as np


FLOAT_RE = re.compile(r"(?<![A-Za-z])[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
TIMESTAMP_RE = re.compile(r"\[(\d+(?:\.\d+)?)\]")
STATE_PREFIX = "Updated state ="
TARGET_OBSERVATION_PREFIX = "Received target observation: position="
TARGET_UPDATE_PREFIX = "Target update:"
COLLISION_PREFIX = "Reached moving target:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot vehicle and target trajectories from ROS 2 logs and mark "
            "the first collision separately from later collisions."
        )
    )
    parser.add_argument(
        "--log",
        required=True,
        type=Path,
        help=(
            "MPC log containing 'Updated state =', target observations, and "
            "'Reached moving target' messages."
        ),
    )
    parser.add_argument(
        "--target-log",
        type=Path,
        default=None,
        help=(
            "Optional moving_target_publisher log containing 'Target update:' "
            "messages. Defaults to --log."
        ),
    )
    parser.add_argument(
        "--save-plot",
        type=Path,
        default=None,
        help="Optional output image path for the 3D plot.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive plot window.",
    )
    return parser.parse_args()


def _extract_floats(text: str) -> list[float]:
    normalized = re.sub(r"np\.float\d+\(", "(", text)
    return [float(value) for value in FLOAT_RE.findall(normalized)]


def _extract_timestamp(line: str) -> float | None:
    matches = TIMESTAMP_RE.findall(line)
    if not matches:
        return None
    return float(matches[-1])


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def _extract_prefixed_chunks(lines: list[str], prefix: str):
    index = 0
    while index < len(lines):
        line = lines[index]
        if prefix not in line:
            index += 1
            continue

        timestamp = _extract_timestamp(line)
        chunk = line.split(prefix, maxsplit=1)[1]
        while chunk.count("[") > chunk.count("]") and index + 1 < len(lines):
            index += 1
            chunk += " " + lines[index].strip()

        yield timestamp, chunk
        index += 1


def parse_vehicle_trajectory(path: Path) -> np.ndarray:
    positions = []
    for _, chunk in _extract_prefixed_chunks(_read_lines(path), STATE_PREFIX):
        values = _extract_floats(chunk)
        if len(values) >= 3:
            positions.append(values[:3])

    if not positions:
        raise ValueError(
            f'No vehicle states found in "{path}". '
            f'Expected messages containing "{STATE_PREFIX}".'
        )
    return np.asarray(positions, dtype=float)


def _position_values(text: str) -> list[float]:
    position_text = text.split("position=", maxsplit=1)[-1]
    for delimiter in (", velocity=", ", radius=", ", target=", ", distance="):
        position_text = position_text.split(delimiter, maxsplit=1)[0]
    return _extract_floats(position_text)


def parse_target_trajectory(path: Path) -> np.ndarray:
    lines = _read_lines(path)
    positions = []

    for _, chunk in _extract_prefixed_chunks(lines, TARGET_OBSERVATION_PREFIX):
        values = _extract_floats(chunk.split(", radius=", maxsplit=1)[0])
        if len(values) >= 3:
            positions.append(values[:3])

    if not positions:
        for _, chunk in _extract_prefixed_chunks(lines, TARGET_UPDATE_PREFIX):
            values = _position_values(chunk)
            if len(values) >= 3:
                positions.append(values[:3])

    if not positions:
        raise ValueError(
            f'No target positions found in "{path}". Expected messages containing '
            f'"{TARGET_OBSERVATION_PREFIX}" or "{TARGET_UPDATE_PREFIX}".'
        )
    return np.asarray(positions, dtype=float)


def parse_collisions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vehicle_positions = []
    target_positions = []
    timestamps = []

    for timestamp, chunk in _extract_prefixed_chunks(
        _read_lines(path), COLLISION_PREFIX
    ):
        position_part, separator, remainder = chunk.partition(", target=")
        if not separator:
            continue

        vehicle_values = _position_values(position_part)
        target_values = _extract_floats(
            remainder.split(", distance=", maxsplit=1)[0]
        )
        if len(vehicle_values) < 3 or len(target_values) < 3:
            continue

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
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    centers = (mins + maxs) / 2.0
    max_range = max(float(np.max(maxs - mins)), 1.0)
    radius = max_range / 2.0
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

    vehicle_plot = ned_to_plot(vehicle)
    target_plot = ned_to_plot(target)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        vehicle_plot[:, 0],
        vehicle_plot[:, 1],
        vehicle_plot[:, 2],
        linewidth=2.0,
        label="Vehicle trajectory",
    )
    ax.plot(
        target_plot[:, 0],
        target_plot[:, 1],
        target_plot[:, 2],
        "--",
        linewidth=2.0,
        label="Target trajectory",
    )

    plot_groups = [vehicle_plot, target_plot]
    if len(collision_vehicle) > 0:
        collision_plot = ned_to_plot(collision_vehicle)
        plot_groups.append(collision_plot)
        ax.scatter(
            collision_plot[0, 0],
            collision_plot[0, 1],
            collision_plot[0, 2],
            marker="*",
            s=220,
            color="red",
            edgecolors="black",
            label="First collision",
            zorder=10,
        )
        if len(collision_plot) > 1:
            ax.scatter(
                collision_plot[1:, 0],
                collision_plot[1:, 1],
                collision_plot[1:, 2],
                marker="x",
                s=80,
                color="darkorange",
                linewidths=2.0,
                label="Later collisions",
                zorder=9,
            )

    set_axes_equal(ax, np.vstack(plot_groups))
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_zlabel("Up [m]")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    target_log = args.log if args.target_log is None else args.target_log

    vehicle = parse_vehicle_trajectory(args.log)
    target = parse_target_trajectory(target_log)
    collision_vehicle, collision_target, collision_timestamps = parse_collisions(
        args.log
    )

    print(f"Vehicle points: {len(vehicle)}")
    print(f"Target points: {len(target)}")
    print(f"Collisions: {len(collision_vehicle)}")
    if len(collision_vehicle) > 0:
        print(f"First collision vehicle position: {collision_vehicle[0]}")
        print(f"First collision target position: {collision_target[0]}")
        if np.isfinite(collision_timestamps[0]):
            print(f"First collision timestamp: {collision_timestamps[0]:.6f}")

    plot_trajectories(
        vehicle=vehicle,
        target=target,
        collision_vehicle=collision_vehicle,
        save_path=args.save_plot,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
