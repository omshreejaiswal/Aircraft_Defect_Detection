from __future__ import annotations

import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

import config

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge


CHART_DPI = 300


@dataclass
class TrainingArtifactStatus:
    source: str
    available: bool


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _save_figure(fig, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=CHART_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(output_path)


def _candidate_artifact_paths() -> list[Path]:
    roots = [config.BASE_DIR, config.MODELS_DIR, config.DATA_DIR]
    names = {
        "training_logs.json",
        "metrics.csv",
        "results.json",
        "history.pkl",
        "training_history.json",
        "results.csv",
    }
    candidates: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.name in names and path not in seen:
                candidates.append(path)
                seen.add(path)
    return sorted(candidates)


def _file_has_content(path: Path) -> bool:
    try:
        if path.suffix == ".json":
            payload = json.loads(path.read_text())
            return bool(payload)
        if path.suffix == ".csv":
            with path.open("r", newline="") as csv_file:
                reader = csv.reader(csv_file)
                rows = list(reader)
            return len(rows) > 1
        if path.suffix == ".pkl":
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            return bool(payload)
    except Exception:
        return False
    return False


def load_training_artifact_status() -> TrainingArtifactStatus:
    for path in _candidate_artifact_paths():
        if _file_has_content(path):
            return TrainingArtifactStatus(source=str(path), available=True)
    return TrainingArtifactStatus(source="No local training history artifact found", available=False)


def _confidence_description(confidence: float) -> str:
    if confidence >= 0.85:
        return "strong"
    if confidence >= 0.60:
        return "moderate"
    return "moderate" if confidence >= 0.45 else "low"


def _occupancy_description(occupancy: float) -> str:
    if occupancy > 40.0:
        return "High"
    if occupancy > 20.0:
        return "Elevated"
    if occupancy > 5.0:
        return "Moderate"
    return "Low"


def _risk_score(result: Any) -> float:
    severity = result.intelligence.get("severity", "Low")
    mapping = {
        "Low": 25.0,
        "Moderate": 55.0,
        "High": 78.0,
        "Critical": 95.0,
    }
    base_score = mapping.get(severity, 25.0)
    occupancy_score = min(100.0, float(result.quantification.get("surface_occupancy", 0.0)) * 2.1)
    confidence_modifier = max(0.0, (float(result.confidence) - 0.5) * 20.0)
    return round(min(100.0, max(base_score, occupancy_score + confidence_modifier)), 1)


def build_reliability_explanations(result: Any, artifact_status: TrainingArtifactStatus | None = None) -> list[str]:
    artifact_status = artifact_status or load_training_artifact_status()
    confidence = float(result.confidence)
    occupancy = float(result.quantification.get("surface_occupancy", 0.0))
    spread = float(result.quantification.get("spread", 0.0))
    risk_score = _risk_score(result)

    lines = []
    if artifact_status.available:
        lines.append(f"Training logs were detected locally at {artifact_status.source}, but this section emphasizes operational reliability indicators from the deployed inspection result.")
    else:
        lines.append("Training metrics unavailable (external model or logs not stored). Instead, operational reliability indicators are provided below.")

    lines.append(
        f"Model confidence is {_confidence_description(confidence)} ({confidence * 100.0:.1f}%), "
        + ("indicating predictions should be verified manually." if confidence < 0.60 else "supporting the current automated assessment.")
    )
    lines.append(
        f"{_occupancy_description(occupancy)} surface occupancy ({occupancy:.1f}%) suggests "
        + ("large-scale structural impact." if occupancy > 40.0 else "localized but meaningful material involvement." if occupancy > 20.0 else "limited visible surface involvement.")
    )
    lines.append(
        f"Defect spread indicator is {spread:.4f}, reflecting the proportion of the inspected mask footprint activated during segmentation."
    )
    lines.append(
        f"Risk score is {risk_score:.1f}/100 based on the current severity logic, confidence, and occupied surface area."
    )
    return lines


def build_reliability_analysis_asset(
    output_dir: Path,
    result: Any,
    history_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    artifact_status = load_training_artifact_status()
    confidence = float(result.confidence) * 100.0
    occupancy = float(result.quantification.get("surface_occupancy", 0.0))
    spread = float(result.quantification.get("spread", 0.0))
    risk_score = _risk_score(result)
    safe_surface = max(0.0, 100.0 - occupancy)
    explanations = build_reliability_explanations(result, artifact_status)

    fig = plt.figure(figsize=(11.5, 7.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[3.2, 1.3], hspace=0.35, wspace=0.28)
    ax_conf = fig.add_subplot(gs[0, 0], aspect="equal")
    ax_pie = fig.add_subplot(gs[0, 1])
    ax_risk = fig.add_subplot(gs[0, 2])
    ax_text = fig.add_subplot(gs[1, :])

    for ax in [ax_conf, ax_risk]:
        ax.axis("off")

    segments = [
        (180, 108, "#DCFCE7"),
        (108, 54, "#FEF3C7"),
        (54, 18, "#FED7AA"),
        (18, 0, "#FECACA"),
    ]
    for start, end, color in segments:
        ax_conf.add_patch(Wedge((0, 0), 1.0, end, start, width=0.26, facecolor=color, edgecolor="white"))
    angle = 180 - (confidence / 100.0) * 180
    x = 0.78 * np.cos(np.deg2rad(angle))
    y = 0.78 * np.sin(np.deg2rad(angle))
    ax_conf.plot([0, x], [0, y], color="#0F172A", linewidth=3)
    ax_conf.scatter([0], [0], color="#0F172A", s=60)
    ax_conf.text(0, -0.14, f"{confidence:.1f}%", ha="center", va="center", fontsize=18, fontweight="bold")
    ax_conf.text(0, -0.32, "Confidence Score", ha="center", va="center", fontsize=11, color="#334155")
    ax_conf.set_xlim(-1.1, 1.1)
    ax_conf.set_ylim(-0.35, 1.1)
    ax_conf.set_title("Confidence Gauge", fontsize=13, pad=12)

    ax_pie.pie(
        [occupancy, safe_surface],
        labels=["Damaged Surface", "Remaining Surface"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#DC2626", "#0F766E"],
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    ax_pie.set_title("Surface Damage Share", fontsize=13, pad=12)

    ax_risk.set_title("Risk Indicator", fontsize=13, pad=12)
    ax_risk.set_xlim(0, 100)
    ax_risk.set_ylim(0, 1)
    ax_risk.axis("off")
    risk_segments = [
        (0, 30, "#DCFCE7", "Low"),
        (30, 60, "#FEF3C7", "Moderate"),
        (60, 85, "#FED7AA", "High"),
        (85, 100, "#FECACA", "Critical"),
    ]
    for start, end, color, label in risk_segments:
        ax_risk.add_patch(Rectangle((start, 0.4), end - start, 0.18, facecolor=color, edgecolor="white"))
        ax_risk.text((start + end) / 2, 0.68, label, ha="center", va="center", fontsize=9, color="#334155")
    ax_risk.plot([risk_score, risk_score], [0.32, 0.68], color="#0F172A", linewidth=3)
    ax_risk.text(50, 0.12, f"Risk Score: {risk_score:.1f}/100", ha="center", va="center", fontsize=12, fontweight="bold")
    ax_risk.text(50, -0.04, f"Spread Indicator: {spread:.4f}", ha="center", va="center", fontsize=11, color="#334155")

    ax_text.axis("off")
    y = 0.95
    for line in explanations:
        ax_text.text(0.0, y, f"- {line}", fontsize=10.5, color="#1E293B", va="top", wrap=True)
        y -= 0.22

    fig.suptitle("Model Reliability Analysis", fontsize=20, fontweight="bold", y=0.98)
    fig.text(0.5, 0.93, "Operational reliability indicators derived from the deployed inspection output.", ha="center", fontsize=11, color="#475569")

    image_path = _save_figure(fig, output_dir / "model_reliability_analysis.png")
    return {
        "title": "Model Reliability Analysis",
        "caption": explanations[0],
        "image_path": image_path,
        "section": "Advanced Analytics",
        "explanations": explanations,
        "artifact_source": artifact_status.source,
    }
