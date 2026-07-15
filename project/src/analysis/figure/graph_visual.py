from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class GraphVisualCue:
    state: Literal["empty", "plotted", "uncertain"]
    trend: Literal["increasing", "decreasing", "horizontal"] | None = None
    confidence: float = 0.0
    variation: Literal["monotonic", "turning", "oscillating"] | None = None
    direction_changes: int = 0
    initial_direction: Literal["increasing", "decreasing", "horizontal"] | None = None
    coordinate_plane: bool = False
    mark_type: Literal["points", "line", "multiple", "unknown"] = "unknown"
    series_count: int | None = None


@dataclass(frozen=True)
class _ColoredPathDetection:
    points: np.ndarray
    mark_type: Literal["points", "line", "multiple"]
    series_count: int


def analyze_graph_visual(image: Image.Image) -> GraphVisualCue:
    """Detect an empty coordinate grid or a strong plotted-line trend cheaply."""
    rgb = np.asarray(image.convert("RGB"))
    if rgb.ndim != 3 or min(rgb.shape[:2]) < 24:
        return GraphVisualCue("uncertain")

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    coordinate_grid, diagonal_lines = _axis_and_diagonal_lines(gray)
    colored = _colored_path_detection(rgb)

    if colored is not None:
        trend, variation, changes, initial = _curve_features(colored.points)
        return GraphVisualCue(
            "plotted",
            trend,
            0.96 if trend else 0.88,
            variation,
            changes,
            initial,
            coordinate_grid,
            colored.mark_type,
            colored.series_count,
        )

    non_axis_points = _largest_non_axis_path(gray)
    if non_axis_points is not None:
        trend, variation, changes, initial = _curve_features(non_axis_points)
        return GraphVisualCue(
            "plotted", trend, 0.84, variation, changes, initial,
            coordinate_grid, "unknown", None,
        )

    if diagonal_lines:
        longest = max(diagonal_lines, key=lambda line: line[4])
        trend = _trend_from_image_slope(longest[5])
        return GraphVisualCue(
            "plotted", trend, 0.82, coordinate_plane=coordinate_grid,
            mark_type="line", series_count=1,
        )

    if coordinate_grid:
        return GraphVisualCue("empty", None, 0.92, coordinate_plane=True)
    return GraphVisualCue("uncertain")


def _colored_path_detection(rgb: np.ndarray) -> _ColoredPathDetection | None:
    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = ((saturation >= 65) & (value >= 45) & (value <= 250)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    minimum_area = max(6, int(height * width * 0.00015))
    candidates: list[tuple[int, int]] = []
    small_components: list[int] = []
    for label in range(1, count):
        x, y, component_width, component_height, area = stats[label]
        span = max(component_width / width, component_height / height)
        if area >= minimum_area and span >= 0.18:
            candidates.append((area, label))
        elif area >= minimum_area and component_width <= width * 0.12 and component_height <= height * 0.12:
            small_components.append(label)
    if candidates:
        candidates.sort(reverse=True)
        selected_labels = [label for _, label in candidates[:6]]
        primary_label = selected_labels[0]
        y_values, x_values = np.where(labels == primary_label)
        mark_type: Literal["line", "multiple"] = "multiple" if len(selected_labels) >= 2 else "line"
        return _ColoredPathDetection(
            np.column_stack((x_values, y_values)).astype(np.float32),
            mark_type,
            len(selected_labels),
        )

    # Scatter plots often contain several disconnected colored point markers.
    # Combine them only when they are distributed in both directions, which
    # avoids treating a horizontal row of toolbar icons as graph data.
    if len(small_components) >= 3:
        combined = np.isin(labels, small_components)
        y_values, x_values = np.where(combined)
        if np.ptp(x_values) >= width * 0.18 and np.ptp(y_values) >= height * 0.12:
            return _ColoredPathDetection(
                np.column_stack((x_values, y_values)).astype(np.float32),
                "points",
                1,
            )
    return None


def _largest_non_axis_path(gray: np.ndarray) -> np.ndarray | None:
    """Recover a long black/gray curve after removing grid and axis strokes."""
    height, width = gray.shape
    edges = cv2.Canny(gray, 50, 150)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, width // 7), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, height // 7)))
    horizontal = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
    axes_and_grid = cv2.dilate(cv2.bitwise_or(horizontal, vertical), np.ones((3, 3), np.uint8))
    residual = cv2.bitwise_and(edges, cv2.bitwise_not(axes_and_grid))
    residual = cv2.morphologyEx(residual, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(residual, connectivity=8)
    minimum_area = max(12, int(height * width * 0.0004))
    candidates: list[tuple[int, int]] = []
    for label in range(1, count):
        _, _, component_width, component_height, area = stats[label]
        if (
            area >= minimum_area
            and component_width >= width * 0.30
            and component_height >= height * 0.15
        ):
            candidates.append((area, label))
    if not candidates:
        return None
    _, label = max(candidates)
    y_values, x_values = np.where(labels == label)
    occupied_ratio = len(np.unique(x_values)) / max(1, int(np.ptp(x_values)) + 1)
    if occupied_ratio < 0.45:
        return None
    return np.column_stack((x_values, y_values)).astype(np.float32)


def _axis_and_diagonal_lines(gray: np.ndarray) -> tuple[bool, list[tuple[int, int, int, int, float, float]]]:
    height, width = gray.shape
    edges = cv2.Canny(gray, 50, 150)
    minimum_length = max(20, int(min(height, width) * 0.24))
    raw_lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(15, int(min(height, width) * 0.10)),
        minLineLength=minimum_length,
        maxLineGap=max(3, int(min(height, width) * 0.03)),
    )
    horizontal = 0
    vertical = 0
    diagonal: list[tuple[int, int, int, int, float, float]] = []
    for raw in raw_lines if raw_lines is not None else []:
        x1, y1, x2, y2 = (int(value) for value in raw[0])
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if not length:
            continue
        angle = abs(math.degrees(math.atan2(dy, dx))) % 180
        if angle <= 7 or angle >= 173:
            horizontal += 1
        elif 83 <= angle <= 97:
            vertical += 1
        elif 14 <= angle <= 166:
            diagonal.append((x1, y1, x2, y2, length, dy / dx if dx else math.inf))
    return horizontal >= 2 and vertical >= 2, diagonal


def _curve_features(
    points: np.ndarray,
) -> tuple[
    Literal["increasing", "decreasing", "horizontal"] | None,
    Literal["monotonic", "turning", "oscillating"] | None,
    int,
    Literal["increasing", "decreasing", "horizontal"] | None,
]:
    x_values, y_values = points[:, 0], points[:, 1]
    if np.ptp(x_values) < 12:
        return None, None, 0, None

    bin_count = max(8, min(36, int(np.ptp(x_values) / 5)))
    edges = np.linspace(float(x_values.min()), float(x_values.max()), bin_count + 1)
    profile_x: list[float] = []
    profile_y: list[float] = []
    for index in range(bin_count):
        upper_inclusive = index == bin_count - 1
        selected = (x_values >= edges[index]) & (
            (x_values <= edges[index + 1]) if upper_inclusive else (x_values < edges[index + 1])
        )
        if np.any(selected):
            profile_x.append(float(np.median(x_values[selected])))
            profile_y.append(float(np.median(y_values[selected])))
    if len(profile_x) < 3:
        slope = float(np.polyfit(x_values, y_values, 1)[0])
        trend = _trend_from_image_slope(slope)
        return trend, "monotonic", 0, trend

    px = np.asarray(profile_x)
    py = np.asarray(profile_y)
    if len(py) >= 5:
        py = np.convolve(np.pad(py, (1, 1), mode="edge"), np.ones(3) / 3, mode="valid")

    amplitude = max(1.0, float(np.ptp(py)))
    differences = np.diff(py)
    noise_floor = max(0.8, amplitude * 0.035)
    signs = [int(np.sign(value)) for value in differences if abs(value) >= noise_floor]
    runs: list[int] = []
    for sign in signs:
        if not runs or sign != runs[-1]:
            runs.append(sign)
    changes = max(0, len(runs) - 1)
    variation: Literal["monotonic", "turning", "oscillating"] = (
        "oscillating" if changes >= 2 else "turning" if changes == 1 else "monotonic"
    )

    trend_x, trend_y = px, py
    if variation == "oscillating" and len(py) >= 9:
        # Average across roughly one oscillation period before fitting the
        # centerline, keeping local waves separate from the overall drift.
        window = max(5, int(round(2 * len(py) / (changes + 1))))
        window = min(window + (window + 1) % 2, len(py) - (len(py) + 1) % 2)
        if window >= 3:
            kernel = np.ones(window) / window
            trend_y = np.convolve(py, kernel, mode="valid")
            offset = (window - 1) // 2
            trend_x = px[offset:offset + len(trend_y)]
    slope = float(np.polyfit(trend_x, trend_y, 1)[0])
    fitted_change = slope * float(np.ptp(px))
    trend = "horizontal" if abs(fitted_change) < max(3.0, amplitude * 0.18) else _trend_from_image_slope(slope)

    initial = _trend_from_image_slope(float(runs[0])) if runs else trend
    return trend, variation, changes, initial


def _trend_from_image_slope(
    slope: float,
) -> Literal["increasing", "decreasing", "horizontal"] | None:
    if not math.isfinite(slope):
        return None
    if abs(slope) < 0.10:
        return "horizontal"
    # Image y increases downward, opposite to mathematical graph coordinates.
    return "increasing" if slope < 0 else "decreasing"
