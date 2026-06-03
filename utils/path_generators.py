#!/usr/bin/env python3

import argparse
import math
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.interpolate import splev, splprep


PathGenerator = Callable[..., np.ndarray]


def _build_z_samples(total_points: int, altitude_m: float, z_ramp_fraction: float) -> np.ndarray:
    ramp_fraction = min(0.5, max(0.01, float(z_ramp_fraction)))
    ramp_points = max(2, int(total_points * ramp_fraction))
    z_samples = np.full(total_points, float(altitude_m), dtype=float)
    s = np.linspace(0.0, 1.0, ramp_points)
    z_samples[:ramp_points] = float(altitude_m) * (3.0 * s**2 - 2.0 * s**3)
    z_samples[0] = 0.0
    return z_samples


def generate_spiral_path(
    radius_m: float = 5.0,
    turns: float = 3.0,
    points_per_turn: int = 80,
    altitude_m: float = -3.0,
    z_ramp_fraction: float = 0.6,
) -> np.ndarray:
    radius_m = max(0.01, float(radius_m))
    turns = max(0.25, float(turns))
    points_per_turn = max(8, int(points_per_turn))

    total_points = max(2, int(math.ceil(turns * points_per_turn)))
    u = np.linspace(0.0, 1.0, total_points)
    theta = 2.0 * math.pi * turns * u

    smooth_u = 3.0 * u**2 - 2.0 * u**3
    radius = radius_m * smooth_u
    x_samples = radius * np.cos(theta)
    y_samples = radius * np.sin(theta)
    z_samples = _build_z_samples(total_points, altitude_m, z_ramp_fraction)

    return np.column_stack((x_samples, y_samples, z_samples))

def generate_square_path(
    side_length_m: float = 2.0,
    altitude_m: float = -1.0,
    points_per_edge: int = 200,
    z_ramp_fraction: float = 0.15,
) -> np.ndarray:
    side_length_m = float(side_length_m)
    points_per_edge = int(points_per_edge)

    corners = [
        (0.0, 0.0),
        (side_length_m, 0.0),
        (side_length_m, side_length_m),
        (0.0, side_length_m),
    ]
    edge_midpoints = [
        (0.5 * side_length_m, 0.0),
        (side_length_m, 0.5 * side_length_m),
        (0.5 * side_length_m, side_length_m),
        (0.0, 0.5 * side_length_m),
    ]

    control_points = np.array(
        [
            corners[0],
            edge_midpoints[0],
            corners[1],
            edge_midpoints[1],
            corners[2],
            edge_midpoints[2],
            corners[3],
            edge_midpoints[3],
        ],
        dtype=float,
    )

    smoothing = 0.05 * side_length_m
    tck, _ = splprep(control_points.T, s=smoothing, per=True)

    total_points = max(2, 4 * points_per_edge)
    u = np.linspace(0.0, 1.0, total_points)
    x_spline, y_spline = splev(u, tck)
    z_samples = _build_z_samples(total_points, altitude_m, z_ramp_fraction)

    return np.column_stack((x_spline, y_spline, z_samples))


def generate_up_path(
    altitude_m: float = -0.5,
    points: int = 120,
) -> np.ndarray:
    points = max(2, int(points))
    u = np.linspace(0.0, 1.0, points)
    smooth_u = 3.0 * u**2 - 2.0 * u**3
    x_samples = np.zeros(points, dtype=float)
    y_samples = np.zeros(points, dtype=float)
    z_samples = float(altitude_m) * smooth_u
    z_samples[0] = 0.0

    return np.column_stack((x_samples, y_samples, z_samples))


def generate_up_diagonal_down_path(
    up_m: float = 0.5,
    diagonal_x_m: float = 0.5,
    diagonal_y_m: float = 0.5,
    down_m: float = 0.45,
    points_per_segment: int = 120,
) -> np.ndarray:
    points_per_segment = max(2, int(points_per_segment))
    up_m = float(up_m)
    diagonal_x_m = float(diagonal_x_m)
    diagonal_y_m = float(diagonal_y_m)
    down_m = float(down_m)

    u = np.linspace(0.0, 1.0, points_per_segment)
    smooth_u = 3.0 * u**2 - 2.0 * u**3

    up_segment = np.column_stack(
        (
            np.zeros(points_per_segment, dtype=float),
            np.zeros(points_per_segment, dtype=float),
            -up_m * smooth_u,
        )
    )
    diagonal_segment = np.column_stack(
        (
            diagonal_x_m * smooth_u,
            diagonal_y_m * smooth_u,
            np.full(points_per_segment, -up_m, dtype=float),
        )
    )
    down_segment = np.column_stack(
        (
            np.full(points_per_segment, diagonal_x_m, dtype=float),
            np.full(points_per_segment, diagonal_y_m, dtype=float),
            -up_m + down_m * smooth_u,
        )
    )

    return np.vstack(
        (
            up_segment,
            diagonal_segment[1:],
            down_segment[1:],
        )
    )

def generate_down_path(
    altitude_m_start: float = -0.5,
    altitude_m_end: float = -0.05,
    points: int = 120,
) -> np.ndarray:
    points = max(2, int(points))
    u = np.linspace(0.0, 1.0, points)
    smooth_u = 3.0 * u**2 - 2.0 * u**3
    x_samples = np.zeros(points, dtype=float)
    y_samples = np.zeros(points, dtype=float)
    z_samples = float(altitude_m_end - altitude_m_start) * smooth_u + altitude_m_start

    return np.column_stack((x_samples, y_samples, z_samples))

def generate_circle_path(
    radius_m: float = 1.0,
    turns: float = 0.8,
    points_per_segment: int = 80,
    altitude_m: float = -0.5,
) -> np.ndarray:
    radius_m = max(0.01, float(radius_m))
    turns = max(0.25, float(turns))
    points_per_segment = max(8, int(points_per_segment))

    u = np.linspace(0.0, 1.0, points_per_segment)
    theta = 2.0 * math.pi * turns * u
    tangent_yaw = theta + 0.5 * math.pi
    start_yaw = tangent_yaw[0]

    up_segment = np.column_stack(
        (
            np.zeros(points_per_segment, dtype=float),
            np.zeros(points_per_segment, dtype=float),
            altitude_m * u,
            np.full(points_per_segment, start_yaw, dtype=float),
        )
    )
    forward_segment = np.column_stack(
        (
            radius_m * u,
            np.full(points_per_segment, 0, dtype=float),
            np.full(points_per_segment, altitude_m, dtype=float),
            np.full(points_per_segment, start_yaw, dtype=float),
        )
    )
    circle_segment = np.column_stack(
        (
            radius_m * np.cos(theta),
            radius_m * np.sin(theta),
            np.full(points_per_segment, altitude_m, dtype=float),
            tangent_yaw,
        )
    )

    return np.vstack(
        (
            up_segment,
            forward_segment[1:],
            circle_segment[1:],
        )
    )

PATH_GENERATORS: dict[str, PathGenerator] = {
    "spiral": generate_spiral_path,
    "square": generate_square_path,
    "up": generate_up_path,
    "up_diagonal_down": generate_up_diagonal_down_path,
    "down" : generate_down_path,
    "circle" : generate_circle_path
}


def save_path(points: np.ndarray, output_file: Path) -> Path:
    output_file = output_file.expanduser()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, np.asarray(points, dtype=float))
    return output_file


def generate_default_paths(output_dir: Path) -> dict[str, Path]:
    return {
        name: save_path(generator(), output_dir / f"{name}_path.npy")
        for name, generator in PATH_GENERATORS.items()
    }


def _parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate built-in trajectory .npy files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory where generated .npy files will be written.",
    )
    return parser.parse_args(args)


def main(args=None) -> None:
    parsed_args = _parse_args(args)
    output_files = generate_default_paths(parsed_args.output_dir)
    for name, output_file in output_files.items():
        points = np.load(output_file, allow_pickle=False)
        print(f"{name}: wrote {len(points)} points to {output_file}")


if __name__ == "__main__":
    main()
