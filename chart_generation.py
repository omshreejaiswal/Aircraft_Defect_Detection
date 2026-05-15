from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

import config
from training_chart_generator import build_reliability_analysis_asset

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge


CHART_DPI = 300


def _save_figure(fig, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=CHART_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(output_path)


def _build_risk_surface_chart(result: Any, output_dir: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.plot([0, 8, 25, 40, 60], [1, 2, 3, 4, 4], color="#1D4ED8", linewidth=2.8, drawstyle="steps-post", label="Risk envelope")
    risk_scale = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
    ax.scatter(result.quantification["surface_occupancy"], risk_scale.get(result.risk, 1), color="#DC2626", s=140, zorder=3, label="Current inspection")
    ax.set_xlabel("Surface Occupancy (%)")
    ax.set_ylabel("Risk Level")
    ax.set_yticks([1, 2, 3, 4], ["Low", "Medium", "High", "Critical"])
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="upper left")
    ax.set_title("Risk vs Surface Damage", loc="center")
    return {
        "title": "Risk vs Surface Damage",
        "caption": "Current inspection risk point plotted against the rule-based surface-damage envelope.",
        "image_path": _save_figure(fig, output_dir / "risk_vs_surface_damage.png"),
        "section": "Advanced Analytics",
    }


def _build_severity_gauge(result: Any, output_dir: Path) -> dict[str, str]:
    value = max(0.0, min(100.0, result.confidence * 100.0))
    fig, ax = plt.subplots(figsize=(8, 4.8), subplot_kw={"aspect": "equal"})
    ax.axis("off")
    segments = [
        (180, 135, "#DCFCE7", "Low"),
        (135, 90, "#FEF3C7", "Moderate"),
        (90, 45, "#FED7AA", "High"),
        (45, 0, "#FECACA", "Critical"),
    ]
    for start, end, color, _ in segments:
        ax.add_patch(Wedge((0, 0), 1.0, end, start, width=0.24, facecolor=color, edgecolor="white"))
    angle = 180 - (value / 100.0) * 180
    x = 0.78 * np.cos(np.deg2rad(angle))
    y = 0.78 * np.sin(np.deg2rad(angle))
    ax.plot([0, x], [0, y], color="#0F172A", linewidth=3)
    ax.scatter([0], [0], color="#0F172A", s=60)
    ax.text(0, -0.15, f"{value:.1f}%", ha="center", va="center", fontsize=18, fontweight="bold")
    ax.text(0, -0.33, result.severity, ha="center", va="center", fontsize=12)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.35, 1.1)
    fig.suptitle("Severity Gauge", y=0.96)
    return {
        "title": "Severity Gauge",
        "caption": "Probability-driven severity gauge using the current inspection confidence.",
        "image_path": _save_figure(fig, output_dir / "severity_gauge.png"),
        "section": "Advanced Analytics",
    }


def _build_distribution_chart(mask: np.ndarray, output_dir: Path) -> dict[str, str]:
    defect_pixels = int(np.count_nonzero(mask > 0))
    monitored = 0
    if defect_pixels > 0:
        kernel = np.ones((25, 25), np.uint8)
        monitored_mask = cv2.dilate((mask > 0).astype(np.uint8) * 255, kernel, iterations=1)
        monitored = int(np.count_nonzero((monitored_mask > 0) & (mask == 0)))
    safe_pixels = max(0, int(mask.size) - defect_pixels - monitored)

    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    ax.pie(
        [defect_pixels, safe_pixels, monitored],
        labels=["Defect", "Safe", "Monitored"],
        autopct="%1.1f%%",
        colors=["#DC2626", "#0F766E", "#F59E0B"],
        startangle=90,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    ax.set_title("Defect Distribution", loc="center")
    return {
        "title": "Defect Distribution",
        "caption": "Relative split between defect pixels, safe surface, and monitored surrounding area.",
        "image_path": _save_figure(fig, output_dir / "defect_distribution.png"),
        "section": "Advanced Analytics",
    }


def _build_heatmap_image(original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(mask, dtype=np.uint8)
    if np.any(mask):
        distance = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
        normalized = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.65, heatmap, 0.35, 0)
    if np.any(mask):
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
    return overlay


def _save_image(image: np.ndarray, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    return str(path)


def build_report_assets(result: Any, history_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    output_dir = config.REPORTS_DIR / "charts" / datetime_safe_stamp()
    heatmap_image = _build_heatmap_image(result.original, result.mask)

    assets = [
        build_reliability_analysis_asset(output_dir, result=result, history_rows=history_rows),
        _build_risk_surface_chart(result, output_dir),
        _build_severity_gauge(result, output_dir),
        _build_distribution_chart(result.mask, output_dir),
        {
            "title": "Defect Area Heatmap",
            "caption": "High-resolution heatmap highlighting the most intense damaged region detected within the inspected surface.",
            "image_path": _save_image(heatmap_image, output_dir / "defect_area_heatmap.png"),
            "section": "Advanced Analytics",
        },
        {
            "title": "Original Inspection Image",
            "caption": "Source aircraft component image used for this inspection run.",
            "image_path": _save_image(result.original, output_dir / "original_input.png"),
            "section": "Visual Evidence",
        },
        {
            "title": "Detection Output",
            "caption": "Detection rendering showing localized defect boxes identified by the model.",
            "image_path": _save_image(result.annotated, output_dir / "detection_output.png"),
            "section": "Visual Evidence",
        },
        {
            "title": "Segmentation Overlay",
            "caption": "Segmentation overlay projected on the inspected component surface.",
            "image_path": _save_image(result.overlay, output_dir / "segmentation_overlay.png"),
            "section": "Visual Evidence",
        },
        {
            "title": "Segmentation Mask",
            "caption": "Binary mask used for defect spread and surface occupancy quantification.",
            "image_path": _save_image(result.mask, output_dir / "segmentation_mask.png"),
            "section": "Visual Evidence",
        },
    ]
    return {"report_images": assets, "heatmap_image": heatmap_image}


def datetime_safe_stamp() -> str:
    from datetime import datetime

    return datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
