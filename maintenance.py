from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np


@dataclass
class MaintenanceAssessment:
    severity: str
    risk: str
    recommendation: str
    action: str
    schedule: str
    priority: str
    maintenance_type: str
    next_inspection_date: str
    notes: list[str]
    insights: dict[str, str]
    component: str
    failure_type: str
    severity_reasoning: str


def _locate_mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    if mask.size == 0 or not np.any(mask):
        return None

    ys, xs = np.where(mask > 0)
    centroid_x = float(np.mean(xs)) / max(1.0, mask.shape[1] - 1)
    centroid_y = float(np.mean(ys)) / max(1.0, mask.shape[0] - 1)
    return centroid_x, centroid_y


def _infer_component(mask: np.ndarray) -> str:
    location = _locate_mask_centroid(mask)
    if location is None:
        return "General Airframe"

    centroid_x, centroid_y = location
    if centroid_x > 0.68:
        return "Tail"
    if centroid_x < 0.30:
        return "Wing"
    if 0.30 <= centroid_x <= 0.68:
        return "Fuselage"
    if centroid_y < 0.33:
        return "Upper Surface"
    return "General Airframe"


def _infer_damage_zone(mask: np.ndarray) -> str:
    location = _locate_mask_centroid(mask)
    if location is None:
        return "general airframe surface"

    centroid_x, centroid_y = location

    horizontal = "left-side" if centroid_x < 0.33 else "central" if centroid_x < 0.66 else "tail-side"
    vertical = "upper" if centroid_y < 0.33 else "mid-body" if centroid_y < 0.66 else "lower"

    if horizontal == "tail-side":
        return f"{vertical} tail section"
    if horizontal == "central":
        return f"{vertical} fuselage section"
    return f"{vertical} wing-edge section"


def _derive_severity(probability: float, occupancy: float, spread: float, detections: int, defect_present: bool) -> str:
    if occupancy > 40.0:
        return "Critical"
    if probability >= 0.78 or occupancy > 25.0 or spread > 0.26:
        return "High"
    if probability >= 0.45 or occupancy > 8.0 or detections > 0:
        return "Moderate"
    if defect_present:
        return "Low"
    return "Low"


def _derive_risk(severity: str, occupancy: float, confidence: float) -> str:
    if severity == "Critical" or occupancy > 45.0:
        return "Critical"
    if severity == "High" or occupancy > 25.0:
        return "High"
    if severity == "Moderate" or confidence < 0.65:
        return "Medium"
    return "Low"


def _schedule_from_severity(severity: str) -> dict[str, str]:
    today = datetime.utcnow()
    if severity == "Critical":
        return {
            "action": "Ground Aircraft",
            "schedule": "Immediate (within 24 hours)",
            "priority": "Priority 1",
            "maintenance_type": "Emergency structural assessment",
            "next_inspection_date": (today + timedelta(days=1)).date().isoformat(),
        }
    if severity == "High":
        return {
            "action": "Urgent Inspection",
            "schedule": "Within 2-3 days",
            "priority": "Priority 2",
            "maintenance_type": "Corrective maintenance",
            "next_inspection_date": (today + timedelta(days=3)).date().isoformat(),
        }
    if severity == "Moderate":
        return {
            "action": "Scheduled Maintenance",
            "schedule": "Within 7 days",
            "priority": "Priority 3",
            "maintenance_type": "Planned repair and monitoring",
            "next_inspection_date": (today + timedelta(days=7)).date().isoformat(),
        }
    return {
        "action": "Routine Check",
        "schedule": "Within 30 days",
        "priority": "Priority 4",
        "maintenance_type": "Routine visual inspection",
        "next_inspection_date": (today + timedelta(days=30)).date().isoformat(),
    }


def _build_insights(
    occupancy: float,
    area: float,
    confidence: float,
    severity: str,
    risk: str,
    zone: str,
) -> dict[str, str]:
    if occupancy > 40.0:
        root_cause = f"Extensive surface damage around the {zone} suggests structural stress concentration, advanced corrosion, or impact-related skin deformation."
    elif occupancy > 20.0:
        root_cause = f"Localized but meaningful damage near the {zone} may indicate repeated vibration loading, fastener fatigue, or early-stage corrosion spread."
    else:
        root_cause = f"Limited anomaly response near the {zone} is consistent with minor coating wear, superficial corrosion, or an isolated surface defect."

    if area > 140000:
        interpretation = "The segmented defect footprint spans a large portion of the inspected region, indicating that the issue is not confined to a single pixel cluster and may affect adjacent structure."
    elif area > 40000:
        interpretation = "The defect footprint is moderate in size and should be treated as a contained but actionable maintenance finding."
    else:
        interpretation = "The damaged area is relatively compact, which supports targeted maintenance rather than broad structural intervention."

    if confidence < 0.60:
        risk_explanation = f"Model confidence is {confidence * 100:.1f}%, so the {risk.lower()}-risk assessment should be confirmed with manual inspection before maintenance release decisions are finalized."
    else:
        risk_explanation = f"The {severity.lower()} severity and {risk.lower()} risk level are supported by both the detector confidence and measured surface occupancy, indicating a credible maintenance signal."

    return {
        "root_cause_suggestion": root_cause,
        "damage_interpretation": interpretation,
        "risk_explanation": risk_explanation,
    }


def _classify_failure_type(occupancy: float, area: float, confidence: float) -> str:
    if occupancy > 32.0 or area > 120000:
        return "Deformation"
    if confidence >= 0.65 and occupancy > 14.0:
        return "Corrosion"
    return "Crack"


def _build_severity_reasoning(severity: str, occupancy: float, confidence: float, detections: int) -> str:
    reasons = []
    if occupancy > 40.0:
        reasons.append(f"surface occupancy reached {occupancy:.1f}% which exceeds the 40% critical threshold")
    elif occupancy > 25.0:
        reasons.append(f"surface occupancy reached {occupancy:.1f}% indicating broad affected area")
    if confidence >= 0.78:
        reasons.append(f"model confidence is strong at {confidence * 100:.1f}%")
    elif confidence < 0.60:
        reasons.append(f"confidence is limited at {confidence * 100:.1f}% and requires manual confirmation")
    if detections > 0:
        reasons.append(f"{detections} localized defect region(s) were detected")
    if not reasons:
        reasons.append("the inspection produced a low-intensity anomaly profile")
    return f"Severity was classified as {severity} because " + ", ".join(reasons) + "."


def assess_maintenance_need(
    probability: float,
    defect_present: bool,
    quantification: dict[str, float],
    detections: int,
    mask: np.ndarray,
) -> MaintenanceAssessment:
    occupancy = float(quantification.get("surface_occupancy", 0.0))
    spread = float(quantification.get("spread", 0.0))
    area = float(quantification.get("area", 0.0))

    severity = _derive_severity(probability, occupancy, spread, detections, defect_present)
    risk = _derive_risk(severity, occupancy, probability)
    schedule_info = _schedule_from_severity(severity)

    notes: list[str] = []
    if probability < 0.60:
        notes.append("Low confidence - manual inspection required.")
    if occupancy > 40.0:
        notes.append("Surface occupancy exceeds 40% and has been escalated to critical severity.")
    if detections == 0 and occupancy > 0.0:
        notes.append("Segmentation indicates possible damage even though the detector found no strong object-level defect boxes.")

    component = _infer_component(mask)
    zone = _infer_damage_zone(mask)
    failure_type = _classify_failure_type(occupancy, area, probability)
    insights = _build_insights(occupancy, area, probability, severity, risk, zone)
    severity_reasoning = _build_severity_reasoning(severity, occupancy, probability, detections)

    return MaintenanceAssessment(
        severity=severity,
        risk=risk,
        recommendation=schedule_info["action"],
        action=schedule_info["action"],
        schedule=schedule_info["schedule"],
        priority=schedule_info["priority"],
        maintenance_type=schedule_info["maintenance_type"],
        next_inspection_date=schedule_info["next_inspection_date"],
        notes=notes,
        insights=insights,
        component=component,
        failure_type=failure_type,
        severity_reasoning=severity_reasoning,
    )


def assessment_to_dict(assessment: MaintenanceAssessment) -> dict[str, Any]:
    return {
        "severity": assessment.severity,
        "risk": assessment.risk,
        "recommendation": assessment.recommendation,
        "action": assessment.action,
        "schedule": assessment.schedule,
        "priority": assessment.priority,
        "maintenance_type": assessment.maintenance_type,
        "next_inspection_date": assessment.next_inspection_date,
        "notes": assessment.notes,
        "insights": assessment.insights,
        "component": assessment.component,
        "failure_type": assessment.failure_type,
        "severity_reasoning": assessment.severity_reasoning,
    }
